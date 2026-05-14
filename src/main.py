#!/usr/bin/env python3
import os, time, urllib.request, socket, json, argparse
import numpy as np
from collections import deque

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task (~10 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")

TIP = [4,8,12,16,20]
PIP = [3,6,10,14,18]
JOINT_LIMITS = [
    (-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
    (-3.0718,-0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973),
]
SMOOTH_WIN = 12

def run(camera_index):
    import cv2, mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def send(cmd): sock.sendto(json.dumps(cmd).encode(), ("127.0.0.1", 5555))

    active_joint   = [None]
    gripper_closed = [False]
    angle_buf      = {i: deque(maxlen=SMOOTH_WIN) for i in range(7)}
    last_result    = [None]
    frame_ts       = [0]

    def fingers_extended(lm):
        ext = [lm[TIP[0]].x < lm[PIP[0]].x]
        for i in range(1,5): ext.append(lm[TIP[i]].y < lm[PIP[i]].y)
        return ext

    def classify(fingers):
        _, index, middle, ring, pinky = fingers
        up = sum([index, middle, ring, pinky])
        if up == 0:                                         return "fist"
        if up >= 4:                                         return "open"
        if index and not middle and not ring and not pinky: return "index"
        if index and middle and not ring and not pinky:     return "peace"
        return "other"

    def wrist_tilt(lm):
        dx = lm[5].x - lm[17].x; dy = lm[5].y - lm[17].y
        return float(np.clip(np.degrees(np.arctan2(dy, dx)), -90, 90))

    def on_result(result, output_image, ts):
        if not result.hand_landmarks: last_result[0]=None; return
        lm = result.hand_landmarks[0]
        fingers = fingers_extended(lm)
        last_result[0] = (classify(fingers), wrist_tilt(lm), lm)

    ensure_model()
    landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM, num_hands=1,
        min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5, result_callback=on_result))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened(): raise RuntimeError(f"Camera {camera_index} unavailable")

    print(f"[OK] Camera {camera_index} open")
    print("☝  Index only     → Joint 1")
    print("✌  Index+middle   → Joint 2")
    print("✊ Fist (J1/J2)   → Joint 3")
    print("↔  Wrist tilt     → Move active joint")
    print("✊ Near cube       → Close gripper")
    print("🖐  Open hand       → Open gripper + deselect")
    print("Q                 → Quit\n")

    def draw_hud(frame, gesture, tilt, joint, gripper):
        cv2.rectangle(frame, (0,0), (380,165), (0,0,0), -1)
        cv2.rectangle(frame, (0,0), (380,165), (60,60,60), 1)
        c = (0,220,100); cr = (0,80,255)
        jname = f"Joint {joint+1}" if joint is not None else "none"
        cv2.putText(frame, f"Gesture : {gesture}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        cv2.putText(frame, f"Active  : {jname}",   (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,200,255) if joint is not None else (100,100,100), 2)
        cv2.putText(frame, f"Tilt    : {tilt:.1f} deg", (10,86), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        cv2.putText(frame, f"Gripper : {'CLOSED' if gripper else 'open'}", (10,114),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cr if gripper else c, 2)
        # Joint dots
        for i in range(7):
            col = (0,200,255) if i==joint else (60,60,60)
            cv2.circle(frame, (20+i*50, 148), 14, col, -1)
            cv2.putText(frame, f"J{i+1}", (10+i*50, 153), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame,1)
            frame_ts[0] += 33
            landmarker.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                frame_ts[0])

            r = last_result[0]
            gesture = r[0] if r else "none"
            tilt    = r[1] if r else 0.0

            if r:
                lm = r[2]

                # Joint selection
                if gesture == "index":
                    if active_joint[0] != 0: print("[Log in] Joint 1")
                    active_joint[0] = 0

                elif gesture == "peace":
                    if active_joint[0] != 1: print("[Log in] Joint 2")
                    active_joint[0] = 1

                elif gesture == "fist":
                    if active_joint[0] in (0, 1):
                        print(f"[Auto] Logged out J{active_joint[0]+1} → Log in Joint 3")
                        active_joint[0] = 2
                    # Gripper close check — hand scale proxy for depth
                    dx = lm[9].x-lm[0].x; dy = lm[9].y-lm[0].y
                    near = (dx**2+dy**2)**0.5 > 0.17
                    if near and not gripper_closed[0]:
                        print("[Gripper] Close")
                        gripper_closed[0] = True
                        send({"type":"gripper","close":True})

                elif gesture == "open":
                    if gripper_closed[0]:
                        print("[Gripper] Open")
                        gripper_closed[0] = False
                        send({"type":"gripper","close":False})
                    if active_joint[0] is not None:
                        print(f"[Log out] Joint {active_joint[0]+1}")
                        active_joint[0] = None

                # Move active joint with wrist tilt + interpolation smoothing
                j = active_joint[0]
                if j is not None and gesture not in ("fist","open"):
                    lo, hi = JOINT_LIMITS[j]
                    t = (tilt + 90.0) / 180.0
                    angle_buf[j].append(lo + t*(hi-lo))
                    send({"type":"joint","joint":j,"angle":float(np.mean(angle_buf[j]))})

            draw_hud(frame, gesture, tilt, active_joint[0], gripper_closed[0])
            cv2.imshow("Team 4 Hand Control — Q to quit", frame)
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
