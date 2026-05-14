#!/usr/bin/env python3
"""
main.py — Full 7-joint Panda control via hand gestures
Joystick-style delta control: tilt = speed + direction, center = stop
"""
import os, socket, json, argparse
import numpy as np
from collections import deque
import urllib.request, time

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
SMOOTH_WIN = 8      # smoothing window for tilt readings
NEAR_THR   = 0.18   # hand scale threshold for gripper close

# Joystick params
DEADZONE   = 8.0    # degrees — tilt within this range = no movement
MAX_TILT   = 45.0   # degrees — full speed at this tilt
SPEED      = 0.03   # radians per frame at full tilt

GESTURE_TO_JOINT = {
    "j1": 0, "j2": 1, "j3": 2, "j4": 3,
    "j5": 4, "j6": 5, "j7": 6,
}
JNAMES = ["J1 shoulder yaw", "J2 shoulder pitch", "J3 elbow",
          "J4 forearm", "J5 wrist pitch", "J6 wrist roll", "J7 hand yaw"]
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


def tilt_to_delta(tilt_deg: float) -> float:
    """
    Joystick-style: dead zone in center, proportional speed outside.
    Returns delta in radians per frame.
    Positive tilt → positive joint direction, negative → negative.
    """
    sign = np.sign(tilt_deg)
    mag  = abs(tilt_deg)
    if mag < DEADZONE:
        return 0.0
    # Normalise from deadzone..max_tilt → 0..1
    t = min((mag - DEADZONE) / (MAX_TILT - DEADZONE), 1.0)
    return sign * t * SPEED


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
    # Current angle per joint — starts at home
    joint_angles   = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    roll_buf       = deque(maxlen=SMOOTH_WIN)
    pitch_buf      = deque(maxlen=SMOOTH_WIN)
    last_result    = [None]
    frame_ts       = [0]

    def fingers_extended(lm):
        ext = [lm[TIP[0]].x < lm[PIP[0]].x]
        for i in range(1, 5):
            ext.append(lm[TIP[i]].y < lm[PIP[i]].y)
        return ext

    def classify(f):
        thumb, index, middle, ring, pinky = f
        if thumb and index and middle and ring and pinky:         return "open"
        if not any(f):                                            return "fist"
        if not thumb and index and middle and ring and pinky:     return "j4"
        if not thumb and index and middle and ring and not pinky: return "j3"
        if not thumb and index and middle and not ring and not pinky: return "j2"
        if not thumb and index and not middle and not ring and not pinky: return "j1"
        if thumb and index and not middle and not ring and pinky: return "j7"
        if thumb and not index and not middle and not ring and pinky: return "j6"
        if thumb and not index and not middle and not ring and not pinky: return "j5"
        return "other"

    def wrist_roll(lm):
        """Left/right tilt — angle of knuckle line, centred at 0."""
        dx = lm[5].x - lm[17].x
        dy = lm[5].y - lm[17].y
        raw = float(np.degrees(np.arctan2(dy, dx)))
        # Centre: hand held flat ≈ 0 degrees
        return float(np.clip(raw, -90, 90))

    def wrist_pitch(lm):
        """Forward/back lean — angle of wrist-to-middle-MCP, centred at 0."""
        dx = lm[9].x - lm[0].x
        dy = lm[9].y - lm[0].y
        raw = float(np.degrees(np.arctan2(dy, dx)))
        # Subtract 90 so pointing camera = ~0
        return float(np.clip(raw - 90, -90, 90))

    def hand_scale(lm):
        dx = lm[9].x - lm[0].x
        dy = lm[9].y - lm[0].y
        return (dx**2 + dy**2) ** 0.5

    def on_result(result, output_image, ts):
        if not result.hand_landmarks:
            last_result[0] = None; return
        lm = result.hand_landmarks[0]
        f  = fingers_extended(lm)
        roll_buf.append(wrist_roll(lm))
        pitch_buf.append(wrist_pitch(lm))
        last_result[0] = {
            "gesture": classify(f),
            "fingers": f,
            "roll":    float(np.mean(roll_buf)),
            "pitch":   float(np.mean(pitch_buf)),
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

    print(f"[OK] Camera {camera_index} — joystick delta control")
    print()
    print("  ☝  1 finger              → J1 shoulder yaw")
    print("  ✌  2 fingers             → J2 shoulder pitch")
    print("  🤟 3 fingers             → J3 elbow")
    print("  🖖 4 fingers (no thumb)  → J4 forearm")
    print("  👍 Thumb only            → J5 wrist pitch")
    print("  🤙 Thumb + pinky         → J6 wrist roll")
    print("  🕷 Thumb+index+pinky     → J7 hand yaw")
    print()
    print("  ↔  Tilt wrist LEFT/RIGHT  → rotate joint that direction")
    print("  ↕  Lean hand FWD/BACK     → also rotates joint")
    print("  Hold flat (centre)        → joint stops")
    print()
    print("  ✊ Fist + hand near cam   → Close gripper")
    print("  🖐 All fingers open       → Open gripper + deselect")
    print()

    def draw_hud(frame, r, roll_delta, pitch_delta):
        g       = r["gesture"] if r else "none"
        roll    = r["roll"]    if r else 0.0
        pitch   = r["pitch"]  if r else 0.0
        scale   = r["scale"]  if r else 0.0
        joint   = active_joint[0]
        gripper = gripper_closed[0]
        near    = scale > NEAR_THR

        cv2.rectangle(frame, (0, 0), (470, 235), (10, 10, 10), -1)
        cv2.rectangle(frame, (0, 0), (470, 235), (70, 70, 70), 1)

        c  = (0, 220, 100)
        cr = (0, 80, 255)
        cy = (0, 200, 255)
        co = (0, 165, 255)   # orange for moving

        glabel = GESTURE_LABELS.get(g, g)
        jname  = JNAMES[joint] if joint is not None else "NONE — show finger gesture"
        jc     = cy if joint is not None else (120, 120, 120)

        cv2.putText(frame, f"Gesture : {glabel}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)
        cv2.putText(frame, f"Active  : {jname}",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.58, jc, 2)

        # Roll bar — shows direction and magnitude
        roll_d = tilt_to_delta(roll)
        pitch_d = tilt_to_delta(pitch)
        moving = abs(roll_d) > 0 or abs(pitch_d) > 0
        rc = co if moving else c

        cv2.putText(frame, f"Roll  : {roll:+.1f} deg  delta={roll_d:+.4f}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc, 2)
        cv2.putText(frame, f"Pitch : {pitch:+.1f} deg  delta={pitch_d:+.4f}",
                    (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc, 2)

        # Visual tilt bar
        bar_cx = 235; bar_y = 122; bar_w = 200
        cv2.rectangle(frame, (bar_cx - bar_w//2, bar_y - 6),
                      (bar_cx + bar_w//2, bar_y + 6), (50, 50, 50), -1)
        fill = int(np.clip(roll / 90 * bar_w//2, -bar_w//2, bar_w//2))
        if fill > 0:
            cv2.rectangle(frame, (bar_cx, bar_y-5), (bar_cx+fill, bar_y+5), co, -1)
        elif fill < 0:
            cv2.rectangle(frame, (bar_cx+fill, bar_y-5), (bar_cx, bar_y+5), co, -1)
        cv2.line(frame, (bar_cx, bar_y-8), (bar_cx, bar_y+8), (150,150,150), 1)

        cv2.putText(frame, f"Proximity: {'NEAR — fist to grip!' if near else 'far'}",
                    (10, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    cr if near else c, 2)
        cv2.putText(frame, f"Gripper : {'CLOSED' if gripper else 'open'}",
                    (10, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    cr if gripper else c, 2)

        # Joint dots
        for i in range(7):
            col = cy if i == joint else (45, 45, 45)
            cx_ = 16 + i * 62
            cv2.circle(frame, (cx_, 195), 18, col, -1)
            cv2.putText(frame, f"J{i+1}", (cx_-12, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        # Finger dots
        if r:
            for i, (up, fl) in enumerate(zip(r["fingers"], ["T","I","M","R","P"])):
                fc = (0, 220, 100) if up else (60, 60, 60)
                cv2.circle(frame, (16 + i*32, 220), 10, fc, -1)
                cv2.putText(frame, fl, (10+i*32, 224),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,0,0), 1)

    roll_delta  = [0.0]
    pitch_delta = [0.0]

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
                        print(f"[Log in] {JNAMES[j]}  angle={joint_angles[j]:.3f}")
                    active_joint[0] = j

                    # ── JOYSTICK DELTA CONTROL ─────────────────────────────
                    rd = tilt_to_delta(roll)
                    pd = tilt_to_delta(pitch)
                    delta = rd + pd * 0.4   # roll dominant, pitch assist

                    if abs(delta) > 0:
                        lo, hi = JOINT_LIMITS[j]
                        joint_angles[j] = float(
                            np.clip(joint_angles[j] + delta, lo, hi))
                        send({"type": "joint", "joint": j,
                              "angle": joint_angles[j]})

                    roll_delta[0]  = rd
                    pitch_delta[0] = pd

                elif g == "fist":
                    if near and not gripper_closed[0]:
                        print("[Gripper] Close")
                        gripper_closed[0] = True
                        send({"type": "gripper", "close": True})

                elif g == "open":
                    if gripper_closed[0]:
                        print("[Gripper] Open")
                        gripper_closed[0] = False
                        send({"type": "gripper", "close": False})
                    if active_joint[0] is not None:
                        print(f"[Log out] {JNAMES[active_joint[0]]}")
                        active_joint[0] = None
                    roll_delta[0] = pitch_delta[0] = 0.0

            draw_hud(frame, r, roll_delta[0], pitch_delta[0])
            cv2.imshow("Panda — 7 Joint Joystick Control  (Q to quit)", frame)
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
