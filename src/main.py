#!/usr/bin/env python3
"""
main.py — 7-joint Panda control, auto-calibrated joystick
Hold finger gesture, then tilt FROM that neutral position to move.
"""
import os, socket, json, argparse
import numpy as np
from collections import deque
import urllib.request

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

JOINT_LIMITS = [
    (-2.8973,  2.8973), (-1.7628,  1.7628), (-2.8973,  2.8973),
    (-3.0718, -0.0698), (-2.8973,  2.8973), (-0.0175,  3.7525), (-2.8973,  2.8973),
]
JNAMES = ["J1 shoulder yaw","J2 shoulder pitch","J3 elbow",
          "J4 forearm","J5 wrist pitch","J6 wrist roll","J7 hand yaw"]
GESTURE_TO_JOINT = {"j1":0,"j2":1,"j3":2,"j4":3,"j5":4,"j6":5,"j7":6}
GESTURE_LABELS   = {
    "j1":"☝ J1","j2":"✌ J2","j3":"🤟 J3","j4":"🖖 J4",
    "j5":"👍 J5","j6":"🤙 J6","j7":"🕷 J7",
    "fist":"✊ fist","open":"🖐 open","other":"...","none":"none",
}
TIP       = [4,8,12,16,20]
PIP       = [3,6,10,14,18]
NEAR_THR  = 0.18
DEADZONE  = 10.0   # degrees from neutral
MAX_TILT  = 40.0   # degrees = full speed
SPEED     = 0.025  # rad/frame at full tilt

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def run(camera_index):
    import cv2, mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def send(cmd): sock.sendto(json.dumps(cmd).encode(), ("127.0.0.1",5555))

    active_joint   = [None]
    gripper_closed = [False]
    joint_angles   = [0.0,-0.785,0.0,-2.356,0.0,1.571,0.785]
    last_result    = [None]
    frame_ts       = [0]
    roll_buf       = deque(maxlen=8)
    pitch_buf      = deque(maxlen=8)

    # Neutral calibration — set when joint is first selected
    neutral_roll  = [0.0]
    neutral_pitch = [0.0]
    prev_gesture  = [None]

    def fingers_extended(lm):
        ext = [lm[TIP[0]].x < lm[PIP[0]].x]
        for i in range(1,5): ext.append(lm[TIP[i]].y < lm[PIP[i]].y)
        return ext

    def classify(f):
        t,i,m,r,p = f
        if t and i and m and r and p:           return "open"
        if not any(f):                          return "fist"
        if not t and i and m and r and p:       return "j4"
        if not t and i and m and r and not p:   return "j3"
        if not t and i and m and not r and not p: return "j2"
        if not t and i and not m and not r and not p: return "j1"
        if t and i and not m and not r and p:   return "j7"
        if t and not i and not m and not r and p: return "j6"
        if t and not i and not m and not r and not p: return "j5"
        return "other"

    def get_roll(lm):
        dx=lm[5].x-lm[17].x; dy=lm[5].y-lm[17].y
        return float(np.clip(np.degrees(np.arctan2(dy,dx)),-90,90))

    def get_pitch(lm):
        dx=lm[9].x-lm[0].x; dy=lm[9].y-lm[0].y
        return float(np.clip(np.degrees(np.arctan2(dy,dx))-90,-90,90))

    def delta_from_tilt(tilt_offset):
        """tilt_offset = current - neutral. Dead zone in centre."""
        sign = np.sign(tilt_offset)
        mag  = abs(tilt_offset)
        if mag < DEADZONE: return 0.0
        t = min((mag - DEADZONE) / (MAX_TILT - DEADZONE), 1.0)
        return float(sign * t * SPEED)

    def on_result(result, output_image, ts):
        if not result.hand_landmarks: last_result[0]=None; return
        lm = result.hand_landmarks[0]
        f  = fingers_extended(lm)
        roll_buf.append(get_roll(lm))
        pitch_buf.append(get_pitch(lm))
        last_result[0] = {
            "gesture": classify(f), "fingers": f,
            "roll":  float(np.mean(roll_buf)),
            "pitch": float(np.mean(pitch_buf)),
            "scale": ((lm[9].x-lm[0].x)**2+(lm[9].y-lm[0].y)**2)**0.5,
        }

    ensure_model()
    landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM, num_hands=1,
        min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5, result_callback=on_result))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened(): raise RuntimeError(f"Camera {camera_index} unavailable")

    print(f"[OK] Camera {camera_index}")
    print("Select joint with fingers, then tilt wrist to move it.")
    print("Your wrist position when selecting = neutral (zero point).\n")

    def draw_hud(frame, r):
        g       = r["gesture"] if r else "none"
        roll    = r["roll"]    if r else 0.0
        pitch   = r["pitch"]  if r else 0.0
        scale   = r["scale"]  if r else 0.0
        joint   = active_joint[0]
        gripper = gripper_closed[0]
        near    = scale > NEAR_THR

        # Offsets from neutral
        ro = roll  - neutral_roll[0]
        po = pitch - neutral_pitch[0]
        rd = delta_from_tilt(ro)
        pd = delta_from_tilt(po)
        moving = abs(rd) > 0 or abs(pd) > 0

        cv2.rectangle(frame,(0,0),(480,240),(10,10,10),-1)
        cv2.rectangle(frame,(0,0),(480,240),(70,70,70),1)
        c=(0,220,100); cr=(0,80,255); cy=(0,200,255); co=(0,165,255)

        glabel = GESTURE_LABELS.get(g,g)
        jname  = JNAMES[joint] if joint is not None else "NONE"
        jc     = cy if joint is not None else (120,120,120)

        cv2.putText(frame,f"Gesture : {glabel}",(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.65,c,2)
        cv2.putText(frame,f"Active  : {jname}",(10,54),cv2.FONT_HERSHEY_SIMPLEX,0.58,jc,2)
        cv2.putText(frame,f"Neutral roll:{neutral_roll[0]:+.1f}  pitch:{neutral_pitch[0]:+.1f}",
                    (10,78),cv2.FONT_HERSHEY_SIMPLEX,0.48,(150,150,150),1)
        rc = co if moving else c
        cv2.putText(frame,f"Offset  roll:{ro:+.1f}  pitch:{po:+.1f}",
                    (10,100),cv2.FONT_HERSHEY_SIMPLEX,0.55,rc,2)
        cv2.putText(frame,f"Delta   roll:{rd:+.4f}  pitch:{pd:+.4f}",
                    (10,122),cv2.FONT_HERSHEY_SIMPLEX,0.52,rc,2)

        # Tilt bar centred on neutral
        bx=240; by=142; bw=200
        cv2.rectangle(frame,(bx-bw//2,by-5),(bx+bw//2,by+5),(50,50,50),-1)
        fill=int(np.clip(ro/MAX_TILT*(bw//2),-bw//2,bw//2))
        if fill>0: cv2.rectangle(frame,(bx,by-4),(bx+fill,by+4),co,-1)
        elif fill<0: cv2.rectangle(frame,(bx+fill,by-4),(bx,by+4),co,-1)
        cv2.line(frame,(bx,by-8),(bx,by+8),(200,200,200),1)

        cv2.putText(frame,f"Proximity: {'NEAR — fist=grip' if near else 'far'}",
                    (10,162),cv2.FONT_HERSHEY_SIMPLEX,0.52,cr if near else c,2)
        cv2.putText(frame,f"Gripper : {'CLOSED' if gripper else 'open'}",
                    (10,184),cv2.FONT_HERSHEY_SIMPLEX,0.65,cr if gripper else c,2)

        for i in range(7):
            col=cy if i==joint else (45,45,45)
            cx_=16+i*62
            cv2.circle(frame,(cx_,212),16,col,-1)
            cv2.putText(frame,f"J{i+1}",(cx_-10,217),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,0,0),1)
        if r:
            for i,(up,fl) in enumerate(zip(r["fingers"],["T","I","M","R","P"])):
                fc=(0,220,100) if up else (60,60,60)
                cv2.circle(frame,(16+i*30,232),9,fc,-1)
                cv2.putText(frame,fl,(10+i*30,236),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0),1)

    try:
        while True:
            ret,frame=cap.read()
            if not ret: break
            frame=cv2.flip(frame,1); frame_ts[0]+=33
            landmarker.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)), frame_ts[0])

            r=last_result[0]; g=r["gesture"] if r else "none"

            if r:
                roll=r["roll"]; pitch=r["pitch"]
                near=r["scale"]>NEAR_THR

                if g in GESTURE_TO_JOINT:
                    j=GESTURE_TO_JOINT[g]
                    # Calibrate neutral when first selecting this joint
                    if prev_gesture[0] != g:
                        neutral_roll[0]  = roll
                        neutral_pitch[0] = pitch
                        print(f"[Log in] {JNAMES[j]}  neutral=({roll:+.1f},{pitch:+.1f})")
                    active_joint[0]=j
                    prev_gesture[0]=g

                    # Compute offset from neutral
                    ro = roll  - neutral_roll[0]
                    po = pitch - neutral_pitch[0]
                    rd = delta_from_tilt(ro)
                    pd = delta_from_tilt(po)
                    delta = rd + pd * 0.4

                    if abs(delta) > 0:
                        lo,hi=JOINT_LIMITS[j]
                        joint_angles[j]=float(np.clip(joint_angles[j]+delta,lo,hi))
                        send({"type":"joint","joint":j,"angle":joint_angles[j]})

                elif g=="fist":
                    prev_gesture[0]=g
                    if near and not gripper_closed[0]:
                        print("[Gripper] Close")
                        gripper_closed[0]=True
                        send({"type":"gripper","close":True})

                elif g=="open":
                    prev_gesture[0]=g
                    if gripper_closed[0]:
                        print("[Gripper] Open")
                        gripper_closed[0]=False
                        send({"type":"gripper","close":False})
                    if active_joint[0] is not None:
                        print(f"[Log out] {JNAMES[active_joint[0]]}")
                        active_joint[0]=None
                    neutral_roll[0]=neutral_pitch[0]=0.0

                else:
                    prev_gesture[0]=g

            draw_hud(frame,r)
            cv2.imshow("Panda 7-Joint Control — Q to quit",frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release(); cv2.destroyAllWindows(); landmarker.close(); sock.close()
        print("Done.")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--camera",type=int,default=2)
    run(ap.parse_args().camera)
