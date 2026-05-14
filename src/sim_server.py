#!/usr/bin/env python3
"""
sim_server.py — PyBullet GUI + physics server
Handles: {"type":"joint", "joint":0, "angle":1.2}
         {"type":"gripper", "close":true}
"""
import socket, json, time
import pybullet as p, pybullet_data
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(
    cameraDistance=1.5, cameraYaw=45,
    cameraPitch=-30, cameraTargetPosition=[0,0,0.3])
p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0,0,0], useFixedBase=True)

col  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025]*3)
vis  = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025]*3, rgbaColor=[0.9,0.2,0.2,1])
cube = p.createMultiBody(0.1, col, vis, [0.3, 0.0, 0.025])

EE_LINK = 11
arm_ids, finger_ids = [], []
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    if info[2] == p.JOINT_REVOLUTE and b'finger' not in info[1]: arm_ids.append(i)
    elif b'finger' in info[1]: finger_ids.append(i)

home = [0,-0.785,0,-2.356,0,1.571,0.785]
for i,jid in enumerate(arm_ids[:7]): p.resetJointState(robot, jid, home[i])
for jid in arm_ids: p.setJointMotorControl2(robot, jid, p.POSITION_CONTROL, force=87)
for fid in finger_ids: p.setJointMotorControl2(robot, fid, p.POSITION_CONTROL, targetPosition=0.04, force=40)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 5555))
sock.setblocking(False)
print(f"[SimServer] Ready on UDP 5555")
print(f"[SimServer] arm_ids={arm_ids}  fingers={finger_ids}")

step = 0
while True:
    try:
        while True:
            data, _ = sock.recvfrom(4096)
            cmd = json.loads(data)
            if cmd["type"] == "joint":
                j = cmd["joint"]
                if j < len(arm_ids):
                    p.setJointMotorControl2(
                        robot, arm_ids[j], p.POSITION_CONTROL,
                        targetPosition=cmd["angle"],
                        maxVelocity=1.5, force=87)
                    print(f"[Sim] Joint {j+1} → {cmd['angle']:.3f} rad", flush=True)
            elif cmd["type"] == "gripper":
                pos = 0.005 if cmd["close"] else 0.04
                for fid in finger_ids:
                    p.setJointMotorControl2(robot, fid, p.POSITION_CONTROL,
                        targetPosition=pos, maxVelocity=0.5, force=40)
                print(f"[Sim] Gripper → {'closed' if cmd['close'] else 'open'}", flush=True)
    except BlockingIOError:
        pass
    p.stepSimulation()
    step += 1
    if step % 240 == 0:
        ee  = p.getLinkState(robot, EE_LINK)[4]
        cub = p.getBasePositionAndOrientation(cube)[0]
        dist = np.linalg.norm(np.array(ee)-np.array(cub))
        print(f"[Sim] EE→cube: {dist:.3f}m", flush=True)
    time.sleep(1.0/240)
