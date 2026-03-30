# Light Threads Hand Tracker

Real-time hand tracking effect built with Python, MediaPipe, and OpenCV.

The app keeps the webcam feed visible and draws dense light strings between both hands, with colored fingertip clusters inspired by the provided reference images.

## Features

- Real-time hand tracking with MediaPipe
- Thin multi-colored strings between both hands
- Smooth hand-point stabilization for cleaner motion
- Camera-feed output instead of a black background
- Automatic download of the MediaPipe hand landmarker model when needed

## Requirements

- Python 3
- Webcam

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python neon_hand_tracker.py
```

Press `q` or `Esc` to quit.

## Notes

- The `models/` folder is ignored in git because the model file is downloaded automatically by the script.
- If your MediaPipe install exposes the newer Tasks API, the project uses the official hand landmarker model bundle automatically.
