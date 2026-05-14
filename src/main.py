#!/usr/bin/env python3
"""
main.py — Team 4 UF exact control scheme
-----------------------------------------
Step 1: Extend index finger only     → Log in to Joint 1
Step 2: Make a fist                  → Log out J1/J2, auto log in to Joint 3
Step 3: Extend index + middle        → Log in to Joint 2
Step 4: Make a fist                  → Log out J1/J2, auto log in to Joint 3
Step 5: Within 0.2m + make fist      → Close gripper
Step 6: Fully open hand              → Open gripper

Movement: wrist ROLL (left/right tilt) moves active joint angle.
          wrist PITCH (hand lean forward/back) also mapped to movement.
Smoothing: 12-frame rolling average (matches slide interpolation note).
"""
import os, socket, json, argparse
import numpy as np
from collections import deque
import urllib.request

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

# Panda joint limits (radians)
JOINT_LIMITS = [
    (-2.8973,  2.8973),   # J1 shoulder yaw
    (-1.7628,  1.7628),   # J2 shoulder pitch
    (-2.8973,  2.8973),   # J3 elbow
    (-3.0718, -0.0698),   # J4
    (-2.8973,  2.8973),   # J5
    (-0.0175,  3.7525),   # J6
    (-2.8973,  2.8973),   # J7
]

TIP        = [4, 8, 12, 16, 20]   # fingertip landmarks
PIP        = [3, 6, 10, 14, 18]   # PIP joint landmarks
SMOOTH_WIN = 12                    # rolling average window (interpolation)

# Proximity threshold for gripper close (hand scale proxy for 0.2m)
NEAR_THRESHOLD = 0.18


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

    # ── State ─────────────────────────────────────────────────────────────────
    active_joint   = [None]   # currently logged-in joint index (0-based)
    gripper_closed = [False]
    angle_buf      = {i: deque(maxlen=SMOOTH_WIN) for i in range(7)}
    last_result    = [None]
    frame_ts       = [0]
    prev_gesture   = [None]

    # ── Landmark helpers ──────────────────────────────────────────────────────

    def fingers_extended(lm):
        """Returns [thumb, index, middle, ring, pinky] as booleans."""
        ext = [lm[TIP[0]].x < lm[PIP[0]].x]   # thumb: tip left of pip
        for i in range(1, 5):
            ext.append(lm[TIP[i]].y < lm[PIP[i]].y)  # tip above pip = extended
        return ext

    def classify(fingers):
        _, index, middle, ring, pinky = fingers
        up = sum([index, middle, ring, pinky])
        if up == 0:                                           return "fist"
        if up >= 4:                                           return "open"
        if index and not middle and not ring and not pinky:   return "index"   # step 1/3
        if index and middle and not ring and not pinky:       return "peace"   # step 3
        return "other"

    def wrist_roll(lm):
        """
        Roll angle from landmark 5 (index MCP) to 17 (pinky MCP).
        Range: -90 to +90 degrees. Matches glove roll axis.
        """
        dx = lm[5].x - lm[17].x
        dy = lm[5].y - lm[17].y
        return float(np.clip(np.degrees(np.arctan2(dy, dx)), -90, 90))

    def wrist_pitch(lm):
        """
        Pitch angle from wrist (0) to middle MCP (9).
        Positive = hand leaning forward, negative = back.
        Range: -90 to +90 degrees. Matches glove pitch axis.
        """
        dx = lm[9].x - lm[0].x
        dy = lm[9].y - lm[0].y
        return float(np.clip(np.degrees(np.arctan2(dy, dx)) - 90, -90, 90))

    def hand_scale(lm):
        """
        Distance between wrist (0) and middle MCP (9) in normalised coords.
        Larger = hand closer to camera = proxy for approaching object.
        ~0.05 at arm's length, ~0.20+ when very close.
        """
        dx = lm[9].x - lm[0].x
        dy = lm[9].y - lm[0].y
        return (dx**2 + dy**2) ** 0.5

    def on_result(result, output_image, ts):
        if not result.hand_landmarks:
            last_result[0] = None
            return
        lm = result.hand_landmarks[0]
        fingers = fingers_extended(lm)
        last_result[0] = {
            "gesture": classify(fingers),
            "roll":    wrist_roll(lm),
            "pitch":   wrist_pitch(lm),
            "scale":   hand_scale(lm),
            "lm":      lm,
        }

    # ── MediaPipe setup ────────────────────────────────────────────────────────
    ensure_model()
    landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
            result_callback=on_result))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {camera_index} unavailable — "
                           "check with: v4l2-ctl --list-devices")

    print(f"[OK] Camera {camera_index} open")
    print()
    print("  Step 1: ☝  Index finger only    → Log in to Joint 1")
    print("  Step 2: ✊ Fist (while in J1)   → Auto log in to Joint 3")
    print("  Step 3: ✌  Index + middle       → Log in to Joint 2")
    print("  Step 4: ✊ Fist (while in J2)   → Auto log in to Joint 3")
    print("  Step 5: ✊ Fist + hand near cam → Close gripper (~0.2m)")
    print("  Step 6: 🖐  Fully open hand      → Open gripper")
    print("          ↔  Wrist roll/pitch     → Move active joint")
    print("          Q                       → Quit")
    print()

    # ── HUD ───────────────────────────────────────────────────────────────────
    def draw_hud(frame, r):
        gesture = r["gesture"] if r else "none"
        roll    = r["roll"]    if r else 0.0
        pitch   = r["pitch"]  if r else 0.0
        scale   = r["scale"]  if r else 0.0
        joint   = active_joint[0]
        gripper = gripper_closed[0]

        # Dark panel
        cv2.rectangle(frame, (0, 0), (420, 200), (10, 10, 10), -1)
        cv2.rectangle(frame, (0, 0), (420, 200), (70, 70, 70), 1)

        c   = (0, 220, 100)
        cr  = (0, 80, 255)
        cy  = (0, 200, 255)
        cg  = (50, 50, 50)
        jn  = f"Joint {joint+1}" if joint is not None else "NONE (log in first)"
        jc  = cy if joint is not None else (120, 120, 120)

        cv2.putText(frame, f"Gesture : {gesture}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, c, 2)
        cv2.putText(frame, f"Active  : {jn}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.68, jc, 2)
        cv2.putText(frame, f"Roll    : {roll:+.1f} deg",
                    (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.62, c, 2)
        cv2.putText(frame, f"Pitch   : {pitch:+.1f} deg",
                    (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, c, 2)

        near = scale > NEAR_THRESHOLD
        prox_col = (0, 80, 255) if near else c
        cv2.putText(frame, f"Proximity: {'NEAR (<0.2m)' if near else 'far'}",
                    (10, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.58, prox_col, 2)

        gcol = cr if gripper else c
        cv2.putText(frame, f"Gripper : {'CLOSED' if gripper else 'open'}",
                    (10, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.68, gcol, 2)

        # Joint indicator row
        for i in range(7):
            col = cy if i == joint else cg
            cx_ = 22 + i * 56
            cv2.circle(frame, (cx_, 188), 16, col, -1)
            cv2.putText(frame, f"J{i+1}", (cx_-10, 193),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_ts[0] += 33

            landmarker.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                frame_ts[0])

            r       = last_result[0]
            gesture = r["gesture"] if r else "none"

            if r:
                roll  = r["roll"]
                pitch = r["pitch"]
                scale = r["scale"]
                near  = scale > NEAR_THRESHOLD

                # ── Step 1: index only → log in to Joint 1 ────────────────────
                if gesture == "index":
                    if active_joint[0] != 0:
                        print("[Step 1] Log in → Joint 1")
                    active_joint[0] = 0

                # ── Step 3: peace → log in to Joint 2 ─────────────────────────
                elif gesture == "peace":
                    if active_joint[0] != 1:
                        print("[Step 3] Log in → Joint 2")
                    active_joint[0] = 1

                # ── Steps 2/4: fist while in J1 or J2 → auto to Joint 3 ───────
                elif gesture == "fist":
                    if active_joint[0] in (0, 1):
                        print(f"[Step 2/4] Log out J{active_joint[0]+1} "
                              f"→ Auto log in Joint 3")
                        active_joint[0] = 2

                    # ── Step 5: near object + fist → close gripper ─────────────
                    if near and not gripper_closed[0]:
                        print("[Step 5] Near object — Close gripper")
                        gripper_closed[0] = True
                        send({"type": "gripper", "close": True})

                # ── Step 6: fully open → open gripper + log out ────────────────
                elif gesture == "open":
                    if gripper_closed[0]:
                        print("[Step 6] Open gripper")
                        gripper_closed[0] = False
                        send({"type": "gripper", "close": False})
                    if active_joint[0] is not None:
                        print(f"[Log out] Joint {active_joint[0]+1}")
                        active_joint[0] = None

                # ── Movement: roll + pitch drive active joint ──────────────────
                j = active_joint[0]
                if j is not None and gesture not in ("fist", "open"):
                    lo, hi = JOINT_LIMITS[j]

                    # Primary: wrist roll (-90..+90) → full joint range
                    # Secondary: pitch offsets by up to ±15% of range
                    t_roll  = (roll  + 90.0) / 180.0        # 0..1
                    t_pitch = (pitch + 90.0) / 180.0        # 0..1
                    # Blend: 70% roll, 30% pitch (matches IMU behaviour)
                    t = 0.7 * t_roll + 0.3 * t_pitch
                    t = float(np.clip(t, 0.0, 1.0))

                    target = lo + t * (hi - lo)
                    angle_buf[j].append(target)
                    smoothed = float(np.mean(angle_buf[j]))   # interpolation
                    send({"type": "joint", "joint": j, "angle": smoothed})

            draw_hud(frame, r)
            cv2.imshow("Panda Hand Control — Q to quit", frame)
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
    ap = argparse.ArgumentParser(
        description="Team 4 UF hand gesture control for Franka Panda")
    ap.add_argument("--camera", type=int, default=2,
                    help="Camera index (default 2 = Logitech C920)")
    run(ap.parse_args().camera)
