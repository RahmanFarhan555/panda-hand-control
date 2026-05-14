# Panda Robot Arm — Hand Gesture Control

> Control a Franka Panda robot arm in real-time using hand gestures captured from a USB webcam. 
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green?logo=google)
![PyBullet](https://img.shields.io/badge/PyBullet-3.2-orange)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Demo

Control a simulated Franka Panda arm using only your hand in front of a Logitech C920 webcam. MediaPipe detects 21 hand landmarks per frame. Finger gestures select joints. Wrist tilt drives the angle. Pinch picks up objects.

```
Logitech C920 → MediaPipe (21 landmarks) → Gesture Classifier → UDP → PyBullet Panda
```

---

## Gesture Controls

| Gesture | Fingers | Action |
|---|---|---|
| ☝ Point | Index only | Log in to **Joint 1** (shoulder yaw) |
| ✌ Peace | Index + middle | Log in to **Joint 2** (shoulder pitch) |
| ✊ Fist | No fingers up | Auto switch → **Joint 3** (elbow) |
| ↔ Wrist tilt | — | Move the currently active joint |
| ✊ Fist (near cam) | Hand close to camera | **Close gripper** |
| 🖐 Open | All fingers | **Open gripper** + deselect joint |
| Q | — | Quit |

---

## Architecture

```
┌──────────────────────────────┐        ┌──────────────────────────────┐
│         main.py              │        │        sim_server.py          │
│                              │        │                              │
│  Logitech C920 (/dev/video2) │        │  PyBullet GUI                │
│           │                  │  UDP   │  Franka Panda URDF           │
│  MediaPipe HandLandmarker    │──────▶ │  7-DOF position control      │
│  Gesture classifier          │ :5555  │  Gripper control             │
│  Wrist tilt + smoother       │        │  Physics @ 240 Hz            │
│  OpenCV HUD overlay          │        │  Red cube (pick & place)     │
└──────────────────────────────┘        └──────────────────────────────┘
```

Two separate processes communicate over UDP on localhost:5555.
This prevents the EGL/X11 GPU context conflict between MediaPipe and PyBullet.

---

## Quick Start

### Without Docker

```bash
git clone https://github.com/RahmanFarhan555/panda-hand-control.git
cd panda-hand-control
pip install -r requirements.txt
export DISPLAY=:1

# Terminal 1 — start physics simulation
python3 src/sim_server.py

# Terminal 2 — start hand tracking
python3 src/main.py --camera 2
```

> On first run, `hand_landmarker.task` (~10 MB) is downloaded automatically.

### With Docker

```bash
xhost +local:docker
export DISPLAY=:1
docker compose up
```

---

## Project Structure

```
panda-hand-control/
├── src/
│   ├── main.py            # Hand tracking + gesture recognition
│   └── sim_server.py      # PyBullet physics + UDP command server
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Pick and Place

1. Use **Joint 1 + 2** to position the arm above the red cube
2. Switch to **Joint 3** to lower the gripper
3. Bring hand close to camera + make a fist → **gripper closes**
4. Raise and move the arm to target position
5. Open hand → **cube drops**

---

## Implementation Notes

Two processes - MediaPipe and PyBullet both initialise an OpenGL/EGL context. Running both in one process causes SIGABRT. Separating into two processes communicating over UDP solves this completely.

Smoothing - raw wrist tilt jitters ±3–5° per frame. A 12-frame rolling average (deque) reduces this to sub-degree noise without noticeable lag — the same interpolation approach used in the original Team 4 demonstration.

Depth proxy - without a depth camera, gripper-close is triggered by hand scale: the distance between wrist landmark 0 and middle-finger MCP landmark 9 in normalised image coordinates. A larger value means the hand is closer to the camera.

Camera index - on Ubuntu with built-in + USB webcam, use `v4l2-ctl --list-devices` to find your Logitech index (typically `/dev/video2`).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Invalid MIT-MAGIC-COOKIE-1` | `export DISPLAY=:1` (run `who` to confirm display number) |
| `ModuleNotFoundError: mediapipe` | `eval "$(~/miniconda3/bin/conda shell.bash hook)"` |
| `Camera N unavailable` | Run `v4l2-ctl --list-devices` and update `--camera` flag |
| Robot not moving | Confirm sim_server prints `[SimServer] Ready on UDP 5555` |
| Core dump on startup | Run as two separate processes — do not merge into one script |

---

## License

MIT — free to use, modify, and distribute.
