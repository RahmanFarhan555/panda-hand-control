#!/usr/bin/env python3
"""
main.py — Full 7-joint Panda control via hand gestures
-------------------------------------------------------
Joint selection by finger count:
  ☝  1 finger  (index)                    → Joint 1
  ✌  2 fingers (index + middle)           → Joint 2
  🤟 3 fingers (index + middle + ring)    → Joint 3
  🖖 4 fingers (index + middle + ring + pinky) → Joint 4
  👍 Thumb only                           → Joint 5
  🤙 Thumb + pinky                        → Joint 6
  🤙 Thumb + index + pinky               → Joint 7

Movement:
  ↔  Wrist roll  (left/right tilt)  → primary axis
  ↕  Wrist pitch (forward/back)     → secondary axis (blended 70/30)
  12-frame rolling average smoothing

Gripper:
  ✊ Fist + hand close to camera    → Close gripper
  🖐 Fully open hand                → Open gripper + deselect joint
"""
import os, socket, json, argparse
import numpy as np
from collections import deque
import urllib.request

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

JOINT_LIMITS = [
    (-2.8973,  2.8973),   # J1 shoulder yaw
    (-1.7628,  1.7628),   # J2 shoulder pitch
    (-2.8973,  2.8973),   # J3 elbow
    (-3.0718, -0.0698),   # J4 forearm
    (-2.8973,  2.8973),   # J5 wrist pitch
    (-0.0175,  3.7525),   # J6 wrist roll
    (-2.8973,  2.8973),   # J7 hand yaw
]

TIP        = [4, 8, 12, 16, 20]
PIP        = [3, 6, 10, 14, 18]
SMOOTH_WIN = 12
NEAR_THR   = 0.18


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task (~10 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model ready.")


def run(camera_index: int):
    import cv2, mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python.vision import (HandLandmarker,
                                               HandLandmarkerOptions,
                                               RunningMode)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def send(cmd):
        sock.sendto(json.dumps(cmd).encode(), ("127.0.0.1", 5555))

    active_joint   = [None]
    gripper_closed = [False]
    angle_buf      = {i: deque(maxlen=SMOOTH_WIN) for i in range(7)}
    last_result    = [None]
    frame_ts       = [0]

    # ── Landmark helpers ──────────────────────────────────────────────────────

    def fingers_extended(lm):
        """[thumb, index, middle, ring, pinky]"""
        ext = [lm[TIP[0]].x < lm[PIP[0]].x]
        for i in range(1, 5):
            ext.append(lm[TIP[i]].y < lm[PIP[i]].y)
        return ext

    def classify(f):
        """Map finger pattern to gesture string."""
        thumb, index, middle, ring, pinky = f

        # Gripper open — all four fingers up
        if index and middle and ring and pinky and not thumb:
            return "open"

        # Joint selections by finger combination
        # 1 finger
        if index and not middle and not ring and not pinky and not thumb:
            return "j1"
        # 2 fingers
        if index and middle and not ring and not pinky and not thumb:
            return "j2"
        # 3 fingers
        if index and middle and ring and not pinky and not thumb:
            return "j3"
        # 4 fingers (index to pinky, no thumb)
        if index and middle and ring and pinky and not thumb:
            return "open"   # already caught above — full open
        # Thumb only
        if thumb and not index and not middle and not ring and not pinky:
            return "j5"
        # Thumb + pinky (shaka)
        if thumb and not index and not middle and not ring and pinky:
            return "j6"
        # Thumb + index + pinky
        if thumb and index and not middle and not ring and pinky:
            return "j7"
        # 4 fingers including pinky but not ring (index+middle+pinky)
        if index and middle and not ring and pinky and not thumb:
            return "j4_alt"
        # All five — open hand
        if thumb and index and middle and ring and pinky:
            return "open"
        # No fingers
        if not any(f):
            return "fist"

        return "other"

    GESTURE_TO_JOINT = {
        "j1": 0, "j2": 1, "j3": 2,
        "j4": 3, "j5": 4, "j6": 5, "j7": 6,
    }

    def wrist_roll(lm):
        dx = lm[5].x - lm[17].x
        dy = lm[5].y - lm[17].y
        return float(np.clip(np.degrees(np.arctan2(dy, dx)), -90, 90))

    def wrist_pitch(lm):
        dx = lm[9].x - lm[0].x
        dy = lm[9].y - lm[0].y
        return float(np.clip(np.degrees(np.arctan2(dy, dx)) - 90, -90, 90))

    def hand_scale(lm):
        dx = lm[9].x - lm[0].x
        dy = lm[9].y - lm[0].y
        return (dx**2 + dy**2) ** 0.5

    def on_result(result, output_image, ts):
        if not result.hand_landmarks:
            last_result[0] = None; return
        lm = result.hand_landmarks[0]
        f  = fingers_extended(lm)
        last_result[0] = {
            "gesture": classify(f),
            "fingers": f,
            "roll":    wrist_roll(lm),
            "pitch":   wrist_pitch(lm),
            "scale":   hand_scale(lm),
        }

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    ensure_model()
    landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.LIVE_STREAM, num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
            result_callback=on_result))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {camera_index} unavailable")

    print(f"[OK] Camera {camera_index} open")
    print()
    print("  ☝  1 finger                  → Joint 1 (shoulder yaw)")
    print("  ✌  2 fingers                 → Joint 2 (shoulder pitch)")
    print("  🤟 3 fingers                 → Joint 3 (elbow)")
    print("  🖖 4 fingers (no thumb)      → Joint 4 (forearm)")
    print("  👍 Thumb only                → Joint 5 (wrist pitch)")
    print("  🤙 Thumb + pinky             → Joint 6 (wrist roll)")
    print("  🤙 Thumb + index + pinky     → Joint 7 (hand yaw)")
    print("  ↔  Wrist roll/pitch          → Move active joint")
    print("  ✊ Fist + hand near cam      → Close gripper")
    print("  🖐  All fingers open          → Open gripper + deselect")
    print("  Q                            → Quit")
    print()

    # Joint names for display
    JNAMES = [
        "J1 shoulder yaw",
        "J2 shoulder pitch",
        "J3 elbow",
        "J4 forearm",
        "J5 wrist pitch",
        "J6 wrist roll",
        "J7 hand yaw",
    ]

    GESTURE_LABELS = {
        "j1": "☝ J1", "j2": "✌ J2", "j3": "🤟 J3",
        "j4": "🖖 J4", "j5": "👍 J5", "j6": "🤙 J6",
        "j7": "🤙+ J7",
        "fist": "✊ fist", "open": "🖐 open", "other": "...",
    }

    def draw_hud(frame, r):
        g      = r["gesture"] if r else "none"
        roll   = r["roll"]    if r else 0.0
        pitch  = r["pitch"]   if r else 0.0
        scale  = r["scale"]   if r else 0.0
        joint  = active_joint[0]
        gripper = gripper_closed[0]
        near   = scale > NEAR_THR

        cv2.rectangle(frame, (0, 0), (440, 215), (10, 10, 10), -1)
        cv2.rectangle(frame, (0, 0), (440, 215), (70, 70, 70), 1)

        c  = (0, 220, 100)
        cr = (0, 80, 255)
        cy = (0, 200, 255)

        glabel = GESTURE_LABELS.get(g, g)
        jname  = JNAMES[joint] if joint is not None else "NONE — show finger gesture"
        jc     = cy if joint is not None else (120, 120, 120)

        cv2.putText(frame, f"Gesture : {glabel}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)
        cv2.putText(frame, f"Active  : {jname}",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.58, jc, 2)
        cv2.putText(frame, f"Roll    : {roll:+.1f}   Pitch: {pitch:+.1f}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.58, c, 2)
        cv2.putText(frame, f"Proximity: {'NEAR — fist to grip!' if near else 'far'}",
                    (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 80, 255) if near else c, 2)
        cv2.putText(frame, f"Gripper : {'CLOSED' if gripper else 'open'}",
                    (10, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    cr if gripper else c, 2)

        # 7 joint dots
        for i in range(7):
            col = cy if i == joint else (45, 45, 45)
            cx_ = 16 + i * 60
            cv2.circle(frame, (cx_, 160), 18, col, -1)
            cv2.putText(frame, f"J{i+1}", (cx_-12, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Finger indicators
        if r:
            fingers = r["fingers"]
            flabels = ["T", "I", "M", "R", "P"]
            for i, (up, fl) in enumerate(zip(fingers, flabels)):
                fc = (0, 220, 100) if up else (60, 60, 60)
                cv2.circle(frame, (16 + i * 30, 195), 10, fc, -1)
                cv2.putText(frame, fl, (10 + i * 30, 199),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            frame_ts[0] += 33

            landmarker.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                frame_ts[0])

            r = last_result[0]
            g = r["gesture"] if r else "none"

            if r:
                scale = r["scale"]
                near  = scale > NEAR_THR
                roll  = r["roll"]
                pitch = r["pitch"]

                # Joint selection
                if g in GESTURE_TO_JOINT:
                    j = GESTURE_TO_JOINT[g]
                    if active_joint[0] != j:
                        print(f"[Log in] {JNAMES[j]}")
                    active_joint[0] = j

                # Gripper close — fist + near
                elif g == "fist":
                    if near and not gripper_closed[0]:
                        print("[Gripper] Close")
                        gripper_closed[0] = True
                        send({"type": "gripper", "close": True})

                # Open hand — gripper open + deselect
                elif g == "open":
                    if gripper_closed[0]:
                        print("[Gripper] Open")
                        gripper_closed[0] = False
                        send({"type": "gripper", "close": False})
                    if active_joint[0] is not None:
                        print(f"[Log out] {JNAMES[active_joint[0]]}")
                        active_joint[0] = None

                # Move active joint
                j = active_joint[0]
                if j is not None and g not in ("fist", "open", "other"):
                    lo, hi = JOINT_LIMITS[j]
                    t = float(np.clip(
                        0.7 * (roll + 90) / 180 + 0.3 * (pitch + 90) / 180,
                        0.0, 1.0))
                    angle_buf[j].append(lo + t * (hi - lo))
                    send({"type": "joint", "joint": j,
                          "angle": float(np.mean(angle_buf[j]))})

            draw_hud(frame, r)
            cv2.imshow("Panda — 7 Joint Hand Control  (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        sock.close()
        print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=2)
    run(ap.parse_args().camera)
