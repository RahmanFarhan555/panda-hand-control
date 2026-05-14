# RoboGrasp — Real-Time Perception & Manipulation
## Phase 1: Workspace, Robot URDF & Gazebo Simulation

---

## System Requirements

| Component | Version |
|-----------|---------|
| Ubuntu | 22.04 LTS |
| ROS2 | Humble Hawksbill |
| Gazebo | Fortress (via ros-humble-gazebo-*) |
| Python | 3.10+ |
| MoveIt2 | 2.x (Humble) |

---

## Step 1 — Install ROS2 Humble (if not done)

```bash
# Set locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS2 apt repo
sudo apt install software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 desktop (includes RViz2, rqt, demos)
sudo apt update && sudo apt upgrade
sudo apt install ros-humble-desktop

# Auto-source in .bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## Step 2 — Install All Project Dependencies

```bash
# Core robotics packages
sudo apt install -y \
  ros-humble-moveit \
  ros-humble-moveit-ros-planning-interface \
  ros-humble-moveit-visual-tools \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-joint-state-publisher \
  ros-humble-joint-state-publisher-gui \
  ros-humble-robot-state-publisher \
  ros-humble-xacro \
  ros-humble-tf2-tools \
  ros-humble-tf2-ros \
  ros-humble-rviz2 \
  ros-humble-rqt \
  ros-humble-rqt-graph \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool

# Python ML/perception stack
pip3 install \
  numpy \
  opencv-python \
  open3d \
  torch torchvision \
  ultralytics \
  scipy \
  transforms3d \
  pyrealsense2 \
  matplotlib \
  pandas

# Initialize rosdep
sudo rosdep init
rosdep update
```

---

## Step 3 — Create the ROS2 Workspace

```bash
# Create workspace
mkdir -p ~/robograsp_ws/src
cd ~/robograsp_ws/src

# Clone Allegro Hand URDF + ros2_control config (lightweight, well-documented)
git clone https://github.com/ros-controls/ros2_control_demos.git

# Or use the Shadow Hand (more complex):
# git clone https://github.com/shadow-robot/sr_common.git

# Build the workspace
cd ~/robograsp_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash

# Auto-source in .bashrc
echo "source ~/robograsp_ws/install/setup.bash" >> ~/.bashrc
```

---

## Step 4 — Create the RoboGrasp Package

```bash
cd ~/robograsp_ws/src

# Create our main package
ros2 pkg create robograsp \
  --build-type ament_python \
  --dependencies rclpy std_msgs sensor_msgs geometry_msgs \
    moveit_msgs trajectory_msgs nav_msgs tf2_ros

# Create subdirectory structure inside the package
cd robograsp
mkdir -p \
  robograsp/perception \
  robograsp/control \
  robograsp/planning \
  robograsp/evaluation \
  launch \
  config \
  urdf \
  worlds \
  meshes

# Rebuild after creating package
cd ~/robograsp_ws
colcon build --symlink-install
source install/setup.bash
```

---

## Step 5 — Understand the Package Structure

After setup, your workspace looks like:

```
robograsp_ws/
└── src/
    └── robograsp/
        ├── robograsp/               # Python source
        │   ├── __init__.py
        │   ├── perception/          # Phase 2: camera, detection, pose
        │   │   ├── camera_node.py
        │   │   ├── object_detector.py
        │   │   └── pose_estimator.py
        │   ├── control/             # Phase 4: PID, trajectory, F/T
        │   │   ├── pid_controller.py
        │   │   ├── trajectory_executor.py
        │   │   └── force_controller.py
        │   ├── planning/            # Phase 3: grasp candidates, scoring
        │   │   ├── grasp_planner.py
        │   │   ├── grasp_scorer.py
        │   │   └── collision_checker.py
        │   └── evaluation/          # Phase 5: metrics, dashboard
        │       ├── grasp_evaluator.py
        │       └── metrics_logger.py
        ├── launch/
        │   ├── simulation.launch.py     # Full simulation bringup
        │   ├── perception.launch.py     # Perception stack only
        │   └── full_pipeline.launch.py  # Everything
        ├── config/
        │   ├── robot_params.yaml        # Joint limits, PID gains
        │   ├── moveit_config.yaml       # MoveIt2 planning config
        │   └── camera_params.yaml       # Camera intrinsics
        ├── urdf/
        │   ├── robotic_hand.urdf.xacro  # Our hand description
        │   └── tabletop_scene.urdf      # Table + objects
        ├── worlds/
        │   └── grasp_world.world        # Gazebo world file
        ├── package.xml
        └── setup.py
```

---

## Step 6 — Verify Everything Works

```bash
# Test ROS2 is working
ros2 topic list          # Should show /rosout, /parameter_events
ros2 node list           # Empty initially — that's fine

# Test Gazebo launches
ros2 launch gazebo_ros gazebo.launch.py

# Test RViz2
rviz2

# Test MoveIt2 installation
ros2 launch moveit2_tutorials demo.launch.py   # Shows Panda arm in RViz

# Check all packages found
ros2 pkg list | grep -E "moveit|gazebo|control"
```

If `moveit2_tutorials demo.launch.py` shows a Panda arm you can drag around — your stack is fully working.

---

## What's Next: Phase 2

Once Phase 1 is verified, Phase 2 builds:
1. A simulated depth camera in Gazebo publishing `/camera/depth/image_raw`
2. A ROS2 node that subscribes and runs YOLOv8 inference
3. Point cloud processing with Open3D
4. 6DoF pose estimation of objects on the table
