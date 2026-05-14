#!/usr/bin/env python3
"""Hand tracker — sends commands to sim_server.py via UDP"""
import os, time, urllib.request, socket, json
import numpy as np
from collections import deque

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model..."); urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

TIP=[4,8,12,16,20]; PIP=[3,6,10,14,18]
WS=dict(x=(-0.4,0.4),y=(-0.4,0.4),z=(0.05,0.55))

def run(camera_index):
    import cv2, mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def send(cmd): sock.sendto(json.dumps(cmd).encode(), ("127.0.0.1", 5555))

    last=[None]; bufs=[deque(maxlen=6) for _ in range(3)]
    pos_buf=deque(maxlen=10); gripper_state=[False]

    def on_result(result, output_image, ts):
        if not result.hand_landmarks: last[0]=None; return
        lm=result.hand_landmarks[0]
        dx=lm[9].x-lm[0].x; dy=lm[9].y-lm[0].y
        scale=(dx**2+dy**2)**0.5
        for buf,v in zip(bufs,[lm[0].x,lm[0].y,float(np.clip((scale-0.05)/0.20,0,1))]):
            buf.append(v)
        sx,sy,sz=[float(np.mean(b)) for b in bufs]
        ext=[lm[TIP[0]].x<lm[PIP[0]].x]
        for i in range(1,5): ext.append(lm[TIP[i]].y<lm[PIP[i]].y)
        pd=((lm[4].x-lm[8].x)**2+(lm[4].y-lm[8].y)**2)**0.5
        up=sum(ext[1:])
        g="pinch" if pd<0.06 else ("fist" if up==0 else ("open" if up>=3 else "move"))
        last[0]=(sx,sy,sz,g)

    ensure_model()
    landmarker=HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM, num_hands=1,
        min_hand_detection_confidence=0.65, min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.55, result_callback=on_result))

    cap=cv2.VideoCapture(camera_index)
    if not cap.isOpened(): raise RuntimeError(f"Camera {camera_index} unavailable")

    frame_ts=0
    print(f"[OK] Camera {camera_index} open. Sending to sim_server on UDP 5555")
    print("PINCH=grip | OPEN=release | Q=quit")

    while True:
        ret,frame=cap.read()
        if not ret: break
        frame=cv2.flip(frame,1); frame_ts+=33
        landmarker.detect_async(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)), frame_ts)

        r=last[0]
        if r:
            nx,ny,nz,g=r
            wx,wy,wz=WS['x'],WS['y'],WS['z']
            tx=wx[0]+nx*(wx[1]-wx[0]); ty=wy[1]-ny*(wy[1]-wy[0]); tz=wz[0]+nz*(wz[1]-wz[0])
            pos_buf.append([tx,ty,tz])
            tgt=np.mean(pos_buf,axis=0).tolist()
            send({"type":"move","pos":tgt})
            if g=="pinch" and not gripper_state[0]:
                gripper_state[0]=True; send({"type":"gripper","close":True})
            elif g=="open" and gripper_state[0]:
                gripper_state[0]=False; send({"type":"gripper","close":False})
            c=(0,220,100)
            cv2.putText(frame,f"Gesture: {g}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,c,2)
            cv2.putText(frame,f"X:{nx:.2f} Y:{ny:.2f} Z:{nz:.2f}",(10,58),cv2.FONT_HERSHEY_SIMPLEX,0.6,c,2)
            gcolor=(0,80,255) if gripper_state[0] else c
            cv2.putText(frame,f"Gripper: {'CLOSED' if gripper_state[0] else 'open'}",(10,82),cv2.FONT_HERSHEY_SIMPLEX,0.6,gcolor,2)
            h,w=frame.shape[:2]
            cv2.circle(frame,(int(nx*w),int(ny*h)),10,c,2)

        cv2.imshow("Hand Tracker — Q to quit", frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); cv2.destroyAllWindows(); landmarker.close(); sock.close()
    print("Done.")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--camera",type=int,default=0)
    run(ap.parse_args().camera)
