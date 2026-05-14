# Panda Robot Arm — Hand Gesture Control

> Control all 7 joints of a Franka Panda robot arm in real-time using hand gestures from a USB webcam. No gloves, no controllers - just your hand.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green?logo=google)
![PyBullet](https://img.shields.io/badge/PyBullet-3.2-orange)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Demo

Control all 7 joints of a simulated Franka Panda arm using only your hand in front of a Logitech C920 webcam. MediaPipe detects 21 hand landmarks per frame. Finger combinations select joints. Wrist tilt drives movement like a joystick - tilt left to rotate one way, tilt right to rotate the other, hold flat to stop.

```
Logitech C920 → MediaPipe (21 landmarks) → Gesture Classifier → UDP → PyBullet Panda
```

---

## Gesture Controls

### Joint Selection - finger combinations

| Gesture | Fingers | Joint | Controls |
|---|---|---|---|
| ☝ Point | Index only | **Joint 1** | Shoulder yaw |
| ✌ Peace | Index + middle | **Joint 2** | Shoulder pitch |
| 🤟 Three | Index + middle + ring | **Joint 3** | Elbow |
| 🖖 Four | Index + middle + ring + pinky | **Joint 4** | Forearm rotation |
| 👍 Thumb | Thumb only | **Joint 5** | Wrist pitch |
| 🤙 Shaka | Thumb + pinky | **Joint 6** | Wrist roll |
| 🕷 Spider | Thumb + index + pinky | **Joint 7** | Hand yaw |

### Movement - joystick style

Hold a finger gesture to select a joint, then tilt your wrist to move it:

| Wrist action | Effect |
|---|---|
| Tilt LEFT | Joint rotates in negative direction |
| Tilt RIGHT | Joint rotates in positive direction |
| Hold flat (centre) | Joint stops — dead zone ±8° |
| Lean FORWARD | Additional movement (secondary axis) |
| Lean BACK | Opposite direction |

Movement speed is proportional to tilt angle. The further you tilt, the faster the joint moves. Release back to centre to stop.

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
│  Roll + pitch joystick       │        │  Red cube (pick & place)     │
│  Delta angle accumulator     │        │                              │
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

The camera window shows a live overlay:

- **Gesture** — current detected gesture with emoji label
- **Active** — which joint is currently selected and its name
- **Roll / Pitch** — wrist angles in degrees with live delta values
- **Tilt bar** — visual left/right tilt indicator, orange when moving
- **Proximity** — whether hand is close enough to trigger gripper
- **Gripper** — CLOSED (red) or open (green)
- **J1–J7 dots** — joint row, active joint highlighted in cyan
- **T I M R P dots** — live finger detection (green = extended, grey = closed)

---

## Pick and Place

A red cube spawns at [0.3, 0.0, 0.025] in the simulation:

1. **J1** — rotate base to face the cube
2. **J2** — pitch shoulder to reach forward
3. **J3** — bend elbow to lower arm
4. **J4–J7** — fine-tune wrist orientation over cube
5. Bring hand **close to camera** + make a **fist** → gripper closes
6. **J2/J3** — raise arm with cube
7. **J1** — rotate to target position
8. **Open hand** → gripper releases, cube drops

---

## Implementation Notes

**Joystick delta control** — instead of mapping tilt angle directly to joint angle (which forces one-directional movement), wrist tilt is used as a velocity input. Tilting left/right drives the joint in that direction at a speed proportional to tilt magnitude. A dead zone of ±8° prevents drift when holding still. This gives intuitive bidirectional control matching how a real joystick works.

**Two processes** — MediaPipe initialises an EGL GPU context for TFLite inference. PyBullet also claims an OpenGL context for its GUI. Running both in one process causes SIGABRT. Two processes with UDP communication solves this completely.

**Smoothing** — wrist roll and pitch readings are averaged over an 8-frame rolling window to reduce jitter before the delta is computed.

**Roll + Pitch axes** — wrist roll (landmarks 5→17, left/right tilt) is the primary movement axis. Wrist pitch (landmarks 0→9, forward/back lean) contributes a secondary 40% assist, giving two degrees of freedom for controlling one joint.

**Finger classification** — tip Y vs PIP Y comparison for fingers 2–5, X-axis comparison for thumb (perpendicular anatomy). Combinations of 7 distinct patterns map to all 7 joints.

**Camera index** — on Ubuntu with built-in + Logitech USB webcam:
```bash
v4l2-ctl --list-devices
# Integrated Camera: /dev/video0
# HD Pro Webcam C920: /dev/video2
```

---

## License

MIT — free to use, modify, and distribute.
