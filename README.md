# Setup

- `pip -m venv .venv`
- `pip install -r requirements.txt`
- Extract video frames on 25 fps
- Extract the key (30 frames) that provide hand gesture action
- Gather the mediapipe coordinates from each hand in a keypoints array
- Compile all keyspoints from all frames into one array
- Follow `dataset.json` structure for setup

# Running

- Currently infer and training on `python3 main.py`
- Model saved as `model.pth`

# Pending

- 🟨 Try other model architectures CNN etc.
- 🟨 LIVE parsing of frames for live translation
- 🟨 Setup infer pipeline
