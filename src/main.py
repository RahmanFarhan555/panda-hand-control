#!/usr/bin/env python3
"""
main.py — Full 7-joint Panda control via hand gestures
-------------------------------------------------------
Joint selection by finger pattern:
  ☝  Index only                      → Joint 1 (shoulder yaw)
  ✌  Index + middle                  → Joint 2 (shoulder pitch)
  🤟 Index + middle + ring           → Joint 3 (elbow)
  🖖 Index + middle + ring + pinky   → Joint 4 (forearm)
  👍 Thumb only                      → Joint 5 (wrist pitch)
  🤙 Thumb + pinky                   → Joint 6 (wrist roll)
  🤙 Thumb + index + pinky           → Joint 7 (hand yaw)
  ✊ Fist + hand near camera         → Close gripper
  🖐 Thumb + all four fingers        → Open gripper + deselect
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
    (-2.8973,  2.8973),
    (-1.7628,  1.7628),
    (-2.8973,  2.8973),
    (-3.0718, -0.0698),
    (-2.8973,  2.8973),
    (-0.0175,  3.7525),
    (-2.8973,  2.8973),
]
JNAMES = ["J1 shoulder yaw", "J2 shoulder pitch", "J3 elbow",
          "J4 forearm", "J5 wrist pitch", "J6 wrist roll", "J7 hand yaw"]

TIP        = [4, 8, 12, 16, 20]
PIP        = [3, 6, 10, 14, 18]
SMOOTH_WIN = 12
NEAR_THR   = 0.18

# Maps gesture string → joint index
GESTURE_TO_JOINT = {
    "j1": 0, "j2": 1, "j3": 2, "j4": 3,
    "j5": 4, "j6": 5, "j7": 6,
}

GESTURE_LABELS = {
    "j1": "☝ J1",  "j2": "✌ J2",  "j3": "🤟 J3", "j4": "🖖 J4",
    "j5": "👍 J5", "j6": "🤙 J6", "j7": "🕷 J7",
    "fist": "✊ fist", "open": "🖐 open", "other": "...", "none": "none",
}


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

    def fingers_extended(lm):
        ext = [lm[TIP[0]].x < lm[PIP[0]].x]   # thumb
        for i in range(1, 5):
            ext.append(lm[TIP[i]].y < lm[PIP[i]].y)
        return ext  # [thumb, index, middle, ring, pinky]

    def classify(f):
        thumb, index, middle, ring, pinky = f

        # All 5 fingers = open hand
        if thumb and index and middle and ring and pinky:   return "open"
        # No fingers = fist
        if not any(f):                                      return "fist"

        # Four fingers (no thumb) = J4
        if not thumb and index and middle and ring and pinky: return "j4"
        # Three fingers = J3
        if not thumb and index and middle and ring and not pinky: return "j3"
        # Two fingers = J2
        if not thumb and index and middle and not ring and not pinky: return "j2"
        # One finger (index) = J1
        if not thumb and index and not middle and not ring and not pinky: return "j1"

        # Thumb + index + pinky = J7
        if thumb and index and not middle and not ring and pinky: return "j7"
        # Thumb + pinky = J6
        if thumb and not index and not middle and not ring and pinky: return "j6"
        # Thumb only = J5
        if thumb and not index and not middle and not ring and not pinky: return "j5"

        return "other"

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

    print(f"[OK] Camera {camera_index} open — full 7-joint control")
    print()
    print("  ☝  1 finger                   → J1 shoulder yaw")
    print("  ✌  2 fingers                  → J2 shoulder pitch")
    print("  🤟 3 fingers                  → J3 elbow")
    print("  🖖 4 fingers (no thumb)       → J4 forearm")
    print("  👍 Thumb only                 → J5 wrist pitch")
    print("  🤙 Thumb + pinky              → J6 wrist roll")
    print("  🕷 Thumb + index + pinky      → J7 hand yaw")
    print("  ✊ Fist + hand near cam       → Close gripper")
    print("  🖐 All 5 fingers              → Open gripper + deselect")
    print("  ↔  Wrist roll/pitch           → Move active joint")
    print()

    def draw_hud(frame, r):
        g      = r["gesture"] if r else "none"
        roll   = r["roll"]    if r else 0.0
        pitch  = r["pitch"]   if r else 0.0
        scale  = r["scale"]   if r else 0.0
        joint  = active_joint[0]
        gripper = gripper_closed[0]
        near   = scale > NEAR_THR

        cv2.rectangle(frame, (0, 0), (460, 220), (10, 10, 10), -1)
        cv2.rectangle(frame, (0, 0), (460, 220), (70, 70, 70), 1)

        c  = (0, 220, 100)
        cr = (0, 80, 255)
        cy = (0, 200, 255)

        jname = JNAMES[joint] if joint is not None else "NONE — show finger gesture"
        jc    = cy if joint is not None else (120, 120, 120)
        glabel = GESTURE_LABELS.get(g, g)

        cv2.putText(frame, f"Gesture : {glabel}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)
        cv2.putText(frame, f"Active  : {jname}",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.58, jc, 2)
        cv2.putText(frame, f"Roll: {roll:+.1f}  Pitch: {pitch:+.1f}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.58, c, 2)
        cv2.putText(frame, f"Proximity: {'NEAR — fist to grip!' if near else 'far'}",
                    (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    cr if near else c, 2)
        cv2.putText(frame, f"Gripper : {'CLOSED' if gripper else 'open'}",
                    (10, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    cr if gripper else c, 2)

        # 7 joint dots
        for i in range(7):
            col = cy if i == joint else (45, 45, 45)
            cx_ = 16 + i * 62
            cv2.circle(frame, (cx_, 162), 18, col, -1)
            cv2.putText(frame, f"J{i+1}", (cx_-12, 167),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        # Finger dots: T I M R P
        if r:
            for i, (up, fl) in enumerate(zip(r["fingers"], ["T","I","M","R","P"])):
                fc = (0, 220, 100) if up else (60, 60, 60)
                cx_ = 16 + i * 32
                cv2.circle(frame, (cx_, 200), 11, fc, -1)
                cv2.putText(frame, fl, (cx_-7, 204),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)

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
                roll  = r["roll"]
                pitch = r["pitch"]
                near  = scale > NEAR_THR

                # Joint selection
                if g in GESTURE_TO_JOINT:
                    j = GESTURE_TO_JOINT[g]
                    if active_joint[0] != j:
                        print(f"[Log in] {JNAMES[j]}")
                    active_joint[0] = j

                    # Move joint — runs every frame while gesture held
                    lo, hi = JOINT_LIMITS[j]
                    t = float(np.clip(
                        0.7 * (roll + 90) / 180 + 0.3 * (pitch + 90) / 180,
                        0.0, 1.0))
                    angle_buf[j].append(lo + t * (hi - lo))
                    send({"type": "joint", "joint": j,
                          "angle": float(np.mean(angle_buf[j]))})

                # Fist — gripper close if near
                elif g == "fist":
                    if near and not gripper_closed[0]:
                        print("[Gripper] Close")
                        gripper_closed[0] = True
                        send({"type": "gripper", "close": True})

                # Open — gripper open + deselect
                elif g == "open":
                    if gripper_closed[0]:
                        print("[Gripper] Open")
                        gripper_closed[0] = False
                        send({"type": "gripper", "close": False})
                    if active_joint[0] is not None:
                        print(f"[Log out] {JNAMES[active_joint[0]]}")
                        active_joint[0] = None

            draw_hud(frame, r)
            cv2.imshow("Panda — 7 Joint Control  (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release(); cv2.destroyAllWindows(); landmarker.close(); sock.close()
        print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=2)
    run(ap.parse_args().camera)
