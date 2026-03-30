from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib import request

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np


NODE_IDS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]
FINGERTIP_IDS = [4, 8, 12, 16, 20]
PALM_IDS = [0, 1, 5, 9, 13, 17]
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
LEFT_COLOR = np.array((215.0, 185.0, 255.0), dtype=np.float32)
RIGHT_COLOR = np.array((170.0, 225.0, 255.0), dtype=np.float32)
THREAD_PALETTE = (
    np.array((220.0, 188.0, 255.0), dtype=np.float32),  # pink-white
    np.array((205.0, 210.0, 255.0), dtype=np.float32),  # peach
    np.array((215.0, 236.0, 255.0), dtype=np.float32),  # warm white
    np.array((175.0, 240.0, 255.0), dtype=np.float32),  # pale yellow
)


@dataclass
class HandState:
    previous_positions: dict[int, np.ndarray] = field(default_factory=dict)
    filtered_positions: dict[int, np.ndarray] = field(default_factory=dict)
    velocity_vectors: dict[int, np.ndarray] = field(default_factory=dict)
    smoothed_speeds: dict[int, float] = field(default_factory=dict)
    average_speed: float = 0.0


def side_color(side_key: str) -> np.ndarray:
    if side_key == "display_left":
        return LEFT_COLOR
    if side_key == "display_right":
        return RIGHT_COLOR
    return THREAD_PALETTE[2]


def thread_color(index: int, spread: int, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    base = THREAD_PALETTE[(index + spread) % len(THREAD_PALETTE)]
    accent = THREAD_PALETTE[(index * 2 + 1) % len(THREAD_PALETTE)]
    length = float(np.linalg.norm(end - start))
    length_mix = min(1.0, length / 700.0)
    cycle_mix = 0.18 + (index % 6) * 0.11
    mixed = base * (1.0 - cycle_mix) + accent * cycle_mix
    return mixed * (1.0 - 0.18 * length_mix) + THREAD_PALETTE[2] * (0.18 * length_mix)


def compute_palm_center(points: dict[int, np.ndarray]) -> np.ndarray:
    palm_points = np.array([points[idx] for idx in PALM_IDS], dtype=np.float32)
    return palm_points.mean(axis=0)


def compute_hand_center(points: dict[int, np.ndarray]) -> np.ndarray:
    return np.array([points[idx] for idx in NODE_IDS], dtype=np.float32).mean(axis=0)
def make_line(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    return np.array([start.astype(np.int32), end.astype(np.int32)], dtype=np.int32)


def stabilize_points(hand_state: HandState, points: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    stabilized: dict[int, np.ndarray] = {}

    for node_id, point in points.items():
        previous = hand_state.filtered_positions.get(node_id)
        if previous is None:
            stabilized_point = point.copy()
        else:
            stabilized_point = previous * 0.72 + point * 0.28

        hand_state.filtered_positions[node_id] = stabilized_point.astype(np.float32)
        stabilized[node_id] = hand_state.filtered_positions[node_id].copy()

    return stabilized


def add_glow_polyline(
    glow_layer: np.ndarray,
    core_layer: np.ndarray,
    polyline: np.ndarray,
    color: np.ndarray,
    thickness: int,
    glow_radius: int,
    brightness: float,
) -> None:
    color_value = tuple(float(channel * brightness) for channel in color)
    faint_value = tuple(float(channel * brightness * 0.08) for channel in color)

    if glow_radius > 0:
        cv2.polylines(
            glow_layer,
            [polyline],
            False,
            faint_value,
            thickness + 1,
            lineType=cv2.LINE_AA,
        )
    cv2.polylines(
        core_layer,
        [polyline],
        False,
        color_value,
        thickness,
        lineType=cv2.LINE_AA,
    )


def add_glow_circle(
    glow_layer: np.ndarray,
    core_layer: np.ndarray,
    center: np.ndarray,
    color: np.ndarray,
    radius: int,
    glow_radius: int,
    brightness: float,
) -> None:
    point = tuple(int(v) for v in center)
    color_value = tuple(float(channel * brightness) for channel in color)
    soft_value = tuple(float(channel * brightness * 0.1) for channel in color)

    cv2.circle(glow_layer, point, radius + glow_radius, soft_value, -1, lineType=cv2.LINE_AA)
    cv2.circle(core_layer, point, radius, color_value, -1, lineType=cv2.LINE_AA)


def update_hand_state(hand_state: HandState, points: dict[int, np.ndarray], dt: float) -> None:
    speed_samples: list[float] = []

    for node_id in NODE_IDS:
        position = points[node_id]
        previous = hand_state.previous_positions.get(node_id)

        if previous is None:
            velocity = np.zeros(2, dtype=np.float32)
            speed = 0.0
        else:
            velocity = (position - previous) / max(dt, 1e-4)
            speed = float(np.linalg.norm(velocity))

        previous_speed = hand_state.smoothed_speeds.get(node_id, 0.0)
        smoothed_speed = previous_speed * 0.7 + speed * 0.3
        hand_state.previous_positions[node_id] = position.copy()
        hand_state.velocity_vectors[node_id] = velocity.astype(np.float32)
        hand_state.smoothed_speeds[node_id] = smoothed_speed
        speed_samples.append(smoothed_speed)

    hand_state.average_speed = float(np.mean(speed_samples)) if speed_samples else 0.0


def compose_effect(glow_layer: np.ndarray, core_layer: np.ndarray) -> np.ndarray:
    soft_points = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=2.0, sigmaY=2.0)
    combined = soft_points * 0.28 + core_layer * 1.05
    return np.clip(combined, 0, 255).astype(np.uint8)


def screen_blend(base_frame: np.ndarray, effect: np.ndarray) -> np.ndarray:
    return cv2.add(base_frame, effect)


def ensure_hand_landmarker_model() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return MODEL_PATH

    download_path = MODEL_PATH.with_suffix(".task.part")
    try:
        print(f"Downloading MediaPipe hand landmarker model to {MODEL_PATH}...")
        request.urlretrieve(MODEL_URL, download_path)
        download_path.replace(MODEL_PATH)
    except Exception as exc:
        if download_path.exists():
            download_path.unlink()
        raise RuntimeError(
            "MediaPipe Tasks is installed, but the required hand landmark model "
            "could not be downloaded automatically. "
            f"Download it from {MODEL_URL} and save it to {MODEL_PATH}."
        ) from exc

    return MODEL_PATH


def create_hand_detector() -> object:
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        return mp.solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.6,
        )

    model_path = ensure_hand_landmarker_model()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect_hands(
    detector: object,
    rgb_frame: np.ndarray,
    width: int,
    height: int,
    timestamp_ms: int,
) -> list[dict[str, object]]:
    if hasattr(detector, "process"):
        result = detector.process(rgb_frame)
        if not (result.multi_hand_landmarks and result.multi_handedness):
            return []

        detected_hands = []
        for hand_landmarks, handedness in zip(
            result.multi_hand_landmarks,
            result.multi_handedness,
        ):
            points = {
                idx: np.array((landmark.x * width, landmark.y * height), dtype=np.float32)
                for idx, landmark in enumerate(hand_landmarks.landmark)
            }
            detected_hands.append(
                {
                    "label": handedness.classification[0].label,
                    "points": points,
                }
            )
        return detected_hands

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect_for_video(mp_image, timestamp_ms)
    if not result.hand_landmarks:
        return []

    detected_hands = []
    for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
        if hand_index < len(result.handedness) and result.handedness[hand_index]:
            label = result.handedness[hand_index][0].category_name or f"Hand {hand_index + 1}"
        else:
            label = f"Hand {hand_index + 1}"

        points = {
            idx: np.array((landmark.x * width, landmark.y * height), dtype=np.float32)
            for idx, landmark in enumerate(hand_landmarks)
        }
        detected_hands.append({"label": label, "points": points})

    return detected_hands


def assign_display_slots(
    detected_hands: list[dict[str, object]],
    width: int,
) -> dict[str, dict[int, np.ndarray]]:
    if not detected_hands:
        return {}

    ordered = sorted(
        detected_hands,
        key=lambda item: compute_hand_center(item["points"])[0],  # type: ignore[index]
    )

    assigned: dict[str, dict[int, np.ndarray]] = {}
    if len(ordered) == 1:
        points = ordered[0]["points"]  # type: ignore[index]
        center_x = compute_hand_center(points)[0]
        side_key = "display_left" if center_x < width * 0.5 else "display_right"
        assigned[side_key] = points
        return assigned

    assigned["display_left"] = ordered[0]["points"]  # type: ignore[index]
    assigned["display_right"] = ordered[-1]["points"]  # type: ignore[index]
    return assigned


def build_bundle_pairs(
    left_points: dict[int, np.ndarray],
    right_points: dict[int, np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    left_nodes = sorted((left_points[idx] for idx in NODE_IDS), key=lambda pt: (pt[1], pt[0]))
    right_nodes = sorted((right_points[idx] for idx in NODE_IDS), key=lambda pt: (pt[1], pt[0]))
    pairs: list[tuple[np.ndarray, np.ndarray, int]] = []

    for index, left_point in enumerate(left_nodes):
        for offset in (-2, -1, 0, 1, 2):
            right_index = index + offset
            if 0 <= right_index < len(right_nodes):
                if abs(offset) <= 1 or index % 2 == 0:
                    pairs.append((left_point, right_nodes[right_index], abs(offset)))

    for left_id in FINGERTIP_IDS:
        for right_id in FINGERTIP_IDS:
            pairs.append((left_points[left_id], right_points[right_id], 1))

    return pairs


def draw_hand_cluster(
    glow_layer: np.ndarray,
    core_layer: np.ndarray,
    points: dict[int, np.ndarray],
    hand_state: HandState,
    side_key: str,
) -> None:
    palm_center = compute_palm_center(points)
    palm_color = side_color(side_key)
    palm_brightness = 0.82 + min(1.0, hand_state.average_speed / 1200.0) * 0.25
    add_glow_circle(
        glow_layer,
        core_layer,
        palm_center,
        palm_color,
        radius=3,
        glow_radius=3,
        brightness=palm_brightness,
    )

    for node_id in NODE_IDS:
        point = points[node_id]
        speed = hand_state.smoothed_speeds.get(node_id, 0.0)
        speed_ratio = min(1.0, speed / 1400.0)
        brightness = 0.84 + speed_ratio * 0.22
        base_color = side_color(side_key)
        if side_key == "display_left":
            accent = THREAD_PALETTE[1]
        else:
            accent = THREAD_PALETTE[3]
        accent_mix = 0.08 + (node_id % 5) * 0.04
        color = base_color * (1.0 - accent_mix) + accent * accent_mix
        radius = 4 if node_id in FINGERTIP_IDS else (3 if node_id in PALM_IDS else 2)
        glow_radius = 4 if node_id in FINGERTIP_IDS else 2

        add_glow_circle(
            glow_layer,
            core_layer,
            point,
            color,
            radius=radius,
            glow_radius=glow_radius,
            brightness=brightness,
        )


def draw_interhand_bundle(
    glow_layer: np.ndarray,
    core_layer: np.ndarray,
    left_points: dict[int, np.ndarray],
    right_points: dict[int, np.ndarray],
    left_state: HandState,
    right_state: HandState,
) -> None:
    pairs = build_bundle_pairs(left_points, right_points)
    motion_ratio = min(1.0, (left_state.average_speed + right_state.average_speed) / 1900.0)

    for index, (start, end, spread) in enumerate(pairs):
        color = thread_color(index, spread, start, end)
        brightness = 0.76 + motion_ratio * 0.22 - spread * 0.03
        glow_radius = 0

        add_glow_polyline(
            glow_layer,
            core_layer,
            make_line(start, end),
            color,
            thickness=1,
            glow_radius=glow_radius,
            brightness=brightness,
        )


def main() -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Unable to open the default camera.")

    hand_states: dict[str, HandState] = {}

    with create_hand_detector() as detector:
        previous_time = time.perf_counter()
        start_time = previous_time

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            now = time.perf_counter()
            dt = min(0.05, max(1.0 / 240.0, now - previous_time))
            previous_time = now
            timestamp_ms = int((now - start_time) * 1000.0)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_hands = detect_hands(
                detector,
                rgb_frame,
                width,
                height,
                timestamp_ms,
            )
            assigned_hands = assign_display_slots(detected_hands, width)

            frame_glow = np.zeros((height, width, 3), dtype=np.float32)
            frame_core = np.zeros((height, width, 3), dtype=np.float32)

            for side_key, points in assigned_hands.items():
                state = hand_states.setdefault(side_key, HandState())
                smooth_points = stabilize_points(state, points)
                update_hand_state(state, smooth_points, dt)
                draw_hand_cluster(frame_glow, frame_core, smooth_points, state, side_key)

            if "display_left" in assigned_hands and "display_right" in assigned_hands:
                draw_interhand_bundle(
                    frame_glow,
                    frame_core,
                    hand_states["display_left"].filtered_positions,
                    hand_states["display_right"].filtered_positions,
                    hand_states["display_left"],
                    hand_states["display_right"],
                )

            softened_frame = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.4, sigmaY=1.4)
            base_frame = cv2.addWeighted(frame, 0.9, softened_frame, 0.1, 2.0)
            effect = compose_effect(frame_glow, frame_core)
            output = screen_blend(base_frame, effect)

            cv2.imshow("Light Threads", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
