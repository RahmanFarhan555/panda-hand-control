# Panda Robot Arm — Hand Gesture Control

> Control a Franka Panda robot arm in real-time using hand gestures captured from a USB webcam. No gloves, no controllers — just your hand.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green?logo=google)
![PyBullet](https://img.shields.io/badge/PyBullet-3.2-orange)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Demo

Control all 7 joints of a simulated Franka Panda arm using only your hand in front of a Logitech C920 webcam. MediaPipe detects 21 hand landmarks per frame. Finger combinations select joints. Wrist tilt drives the angle. Pinch picks up a red cube.

```
Logitech C920 → MediaPipe (21 landmarks) → Gesture Classifier → UDP → PyBullet Panda
```

---

## Gesture Controls

### Joint Selection — finger combinations

| Gesture | Fingers | Joint | Controls |
|---|---|---|---|
| ☝ Point | Index only | **Joint 1** | Shoulder yaw |
| ✌ Peace | Index + middle | **Joint 2** | Shoulder pitch |
| 🤟 Three | Index + middle + ring | **Joint 3** | Elbow |
| 🖖 Four | Index + middle + ring + pinky | **Joint 4** | Forearm rotation |
| 👍 Thumb | Thumb only | **Joint 5** | Wrist pitch |
| 🤙 Shaka | Thumb + pinky | **Joint 6** | Wrist roll |
| 🕷 Spider | Thumb + index + pinky | **Joint 7** | Hand yaw |

### Movement

| Action | Effect |
|---|---|
| ↔ Wrist roll (tilt left/right) | Primary movement axis (70% weight) |
| ↕ Wrist pitch (lean forward/back) | Secondary movement axis (30% weight) |
| 12-frame rolling average | Smoothing — removes jitter |

### Gripper

| Gesture | Action |
|---|---|
| ✊ Fist + hand close to camera | Close gripper |
| 🖐 All 5 fingers open | Open gripper + deselect joint |

---

## Architecture

```
┌──────────────────────────────┐        ┌──────────────────────────────┐
│         main.py              │        │        sim_server.py          │
│                              │        │                              │
│  Logitech C920 (/dev/video2) │        │  PyBullet GUI                │
│           │                  │  UDP   │  Franka Panda URDF           │
│  MediaPipe HandLandmarker    │──────▶ │  7-DOF position control      │
│  21 landmark detection       │ :5555  │  Gripper control             │
│  Finger combination classify │        │  Physics @ 240 Hz            │
│  Roll + pitch estimation     │        │  Red cube (pick & place)     │
│  12-frame smoother           │        │                              │
│  OpenCV HUD overlay          │        └──────────────────────────────┘
└──────────────────────────────┘
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

> On first run, `hand_landmarker.task` (~10 MB) is downloaded automatically into `src/`.

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

## HUD Display

The camera window shows a live overlay with:

- **Gesture** — current detected gesture with emoji label
- **Active** — which joint is currently selected
- **Roll / Pitch** — wrist angles in degrees
- **Proximity** — whether hand is close enough to trigger gripper
- **Gripper** — CLOSED (red) or open (green)
- **J1–J7 dots** — joint indicator row, active joint highlighted in blue
- **T I M R P dots** — live finger detection status (green = extended)

---

## Pick and Place

A red cube spawns at [0.3, 0.0, 0.025] in the simulation:

1. Use **J1 + J2** to position the arm above the cube
2. Use **J3** to lower the elbow
3. Use **J4–J7** to fine-tune wrist position over the cube
4. Bring hand **close to camera** + make a **fist** → gripper closes
5. Raise the arm using J2/J3
6. Move to target position
7. **Open hand** → gripper releases, cube drops

---

## Implementation Notes

**Two processes** — MediaPipe initialises an EGL GPU context for TFLite inference. PyBullet also claims an OpenGL context for its GUI. Running both in one process causes SIGABRT. Two processes with UDP communication solves this completely.

**Smoothing** — raw wrist tilt jitters ±3–5° per frame. A 12-frame rolling average (deque) reduces this to sub-degree noise without noticeable lag. This matches the interpolation approach described in the original Team 4 demonstration.

**Roll + Pitch blending** — wrist roll (landmarks 5→17) is the primary, reliable axis. Wrist pitch (landmarks 0→9) adds a secondary movement axis, blended at 70% roll / 30% pitch to mimic a two-axis IMU.

**Depth proxy** — gripper-close is triggered by hand scale (wrist-to-MCP distance in normalised image coordinates). Larger value = hand closer to camera = approaching the object. Threshold tuned to approximately 0.2m equivalent.

**Finger classification** — MediaPipe provides normalised 3D landmark coordinates. Finger extension is detected by comparing tip Y vs PIP Y (tip above knuckle = extended). Thumb uses X-axis comparison due to its perpendicular anatomy.

**Camera index** — on Ubuntu with built-in + Logitech USB webcam:
```bash
v4l2-ctl --list-devices
# Integrated Camera: /dev/video0
# HD Pro Webcam C920: /dev/video2
```

---

## Inspired By

Team 4 demonstration — Herbert Wertheim College of Engineering, University of Florida, which used a **HiWonder wireless IMU glove + ROS2 + WiFi** to control a Panda arm in PyBullet. This project replicates and extends that using only a standard USB webcam and Python.

Key improvements over the original:
- No hardware glove required — standard USB webcam only
- All 7 joints controllable (original demonstrated 3)
- Two-axis wrist control (roll + pitch) via camera geometry
- Pick and place with proximity-based gripper trigger
- Docker support for reproducible deployment

- Robot model: **Franka Emika Panda** (via `pybullet_data`)
- Hand tracking: **Google MediaPipe** Hand Landmarker Task API
- Physics engine: **PyBullet** (Bullet Physics SDK)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Invalid MIT-MAGIC-COOKIE-1` | `export DISPLAY=:1` (run `who` to confirm display number) |
| `ModuleNotFoundError: mediapipe` | `eval "$(~/miniconda3/bin/conda shell.bash hook)"` |
| `Camera N unavailable` | Run `v4l2-ctl --list-devices` and update `--camera` flag |
| `Address already in use` | `kill $(lsof -t -i:5555)` then restart sim_server |
| Robot not moving | Confirm sim_server prints `[SimServer] Ready on UDP 5555` |
| Core dump on startup | Run as two separate processes — do not merge into one script |
| Thumb always detected up | Ensure hand is fully visible and well-lit |

---

## License

MIT — free to use, modify, and distribute.
