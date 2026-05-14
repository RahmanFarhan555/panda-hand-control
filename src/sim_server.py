#!/usr/bin/env python3
"""Runs PyBullet GUI + physics. Receives commands via UDP from main.py"""
import socket, json, pybullet as p, pybullet_data, numpy as np, time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025]*3)
vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025]*3, rgbaColor=[0.9,0.2,0.2,1])
cube = p.createMultiBody(0.1, col, vis, [0.3,0.0,0.025])

arm_ids, finger_ids = [], []
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    if info[2]==p.JOINT_REVOLUTE and b'finger' not in info[1]: arm_ids.append(i)
    elif b'finger' in info[1]: finger_ids.append(i)

home=[0,-0.785,0,-2.356,0,1.571,0.785]
for i,jid in enumerate(arm_ids[:7]): p.resetJointState(robot,jid,home[i])
for jid in arm_ids: p.setJointMotorControl2(robot,jid,p.POSITION_CONTROL,force=87)
for fid in finger_ids: p.setJointMotorControl2(robot,fid,p.POSITION_CONTROL,targetPosition=0.04,force=40)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 5555))
sock.setblocking(False)
print("[SimServer] PyBullet GUI ready, listening on UDP 5555")

EE=11
while True:
    try:
        data, _ = sock.recvfrom(4096)
        cmd = json.loads(data)
        if cmd["type"] == "move":
            orn = p.getQuaternionFromEuler([np.pi,0,0])
            jp = p.calculateInverseKinematics(robot, EE, cmd["pos"], orn,
                                              maxNumIterations=100, residualThreshold=1e-4)
            for i,jid in enumerate(arm_ids):
                if i<len(jp):
                    p.setJointMotorControl2(robot,jid,p.POSITION_CONTROL,
                                            targetPosition=jp[i],maxVelocity=2.0,force=87)
        elif cmd["type"] == "gripper":
            pos=0.005 if cmd["close"] else 0.04
            for fid in finger_ids:
                p.setJointMotorControl2(robot,fid,p.POSITION_CONTROL,
                                        targetPosition=pos,maxVelocity=0.5,force=40)
    except BlockingIOError:
        pass
    p.stepSimulation()
    time.sleep(1.0/240)
