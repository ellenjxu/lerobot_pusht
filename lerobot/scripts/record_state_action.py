"""
Records state-image pairs for Koch robot. Press key to record.

Useful for training encoder and decoder for state space.  E.g. translate the robot DOF to x,y coordinates for push T
"""

## CHANGE THIS
# run `ls /dev | grep ttyACM` to find the port
# flatpak run com.obsproject.Studio then `v4l2-ctl --list-devices` to find the camera index
follower_port = "/dev/ttyACM0"
leader_port = "/dev/ttyACM1"
camera_index = 4 # flatpak run obs studio. 

from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera

import time
import cv2
from lerobot.scripts.control_robot import busy_wait
import os
import sys
import numpy as np

leader_config = DynamixelMotorsBusConfig(
    port=leader_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl330-m077"),
        "shoulder_lift": (2, "xl330-m077"),
        "elbow_flex": (3, "xl330-m077"),
        "wrist_flex": (4, "xl330-m077"),
        "wrist_roll": (5, "xl330-m077"),
        "gripper": (6, "xl330-m077"),
    },
)

follower_config = DynamixelMotorsBusConfig(
    port=follower_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl430-w250"),
        "shoulder_lift": (2, "xl430-w250"),
        "elbow_flex": (3, "xl330-m288"),
        "wrist_flex": (4, "xl330-m288"),
        "wrist_roll": (5, "xl330-m288"),
        "gripper": (6, "xl330-m288"),
    },
)

leader_arm = DynamixelMotorsBus(leader_config)
follower_arm = DynamixelMotorsBus(follower_config)

robot_config = KochRobotConfig(
    leader_arms={"main": leader_config},
    follower_arms={"main": follower_config},
    calibration_dir=".cache/calibration/koch",
    cameras={
        # "laptop": OpenCVCameraConfig(0, fps=30, width=640, height=480),
        "phone": OpenCVCameraConfig(camera_index=camera_index, fps=30, width=640, height=480),
    },
    # cameras={},
)
robot = ManipulatorRobot(robot_config)
robot.connect()

### MAIN CODE FOR RECORDING DATA ###

# record_time_s = 60
fps = 60

states = []
actions = []
frames = []

print("Press any key to record a state-image pair. Press 'q' to quit.")

# Create a window to capture key presses
cv2.namedWindow("Key Press Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Key Press Window", 300, 100)

running = True
ts = 0
while running:
    start_time = time.perf_counter()
    observation, action = robot.teleop_step(record_data=True)
    
    # Display the latest frame if available
    if "observation.images.phone" in observation:
        frame = observation["observation.images.phone"].numpy().astype(np.uint8)
        cv2.imshow("Key Press Window", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Check for key press (non-blocking)
    key = cv2.waitKey(1)
    if key != -1:
        if key == ord('q'):
            print("Quitting...")
            running = False
        else:
            print(f"Recording state-image pair #{len(frames)}")
            frames.append(observation["observation.images.phone"])
            states.append(observation["observation.state"])
            actions.append(action["action"])

    ts += 1
    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)

cv2.destroyAllWindows()

# Note that observation and action are available in RAM, but
# you could potentially store them on disk with pickle/hdf5 or
# our optimized format `LeRobotDataset`. More on this next.

output_dir = "outputs/robot_cache"
os.makedirs(output_dir, exist_ok=True)

# save each (current state, frame)
# current state is saved in .txt file
# frames are stored as idx.png
import cv2
import numpy as np
import pickle as pkl

with open(f"{output_dir}/states.pkl", "wb") as f:
    pkl.dump(states, f)

with open(f"{output_dir}/actions.pkl", "wb") as f:
    pkl.dump(actions, f)

for i in range(len(frames)):
    frame = frames[i].numpy().astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f"{output_dir}/{i}.png", frame)

print(f"Saved {len(frames)} state-image pairs to {output_dir}")