"""
sim2real on Lerobot.
"""

# TODO: move to setup helper
## CHANGE THIS
# run `ls /dev | grep ttyACM` to find the port
# flatpak run com.obsproject.Studio then `v4l2-ctl --list-devices` to find the camera index
follower_port = "/dev/ttyACM0"
leader_port = "/dev/ttyACM1"
camera_index = 4 # flatpak run obs studio. 

import os
import sys
sys.path.append('/home/ellen/code/*/lerobot_pusht')

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

##################################################################################3

# modified from 02_teleop.ipynb

inference_time_s = 20 # 20s for rollout
fps = 30
device = "cuda"

s_total = time.perf_counter()
frames = []
log_data = []

import torch
from nathan.get_t_info.mask import get_t_position_and_orientation
from helper import get_dataset, forward_kinematics, inverse_kinematics, coordinate_transform
from run_pusht import load_policy_from_checkpoint, policy_step

# optional: start the env for render
# import matplotlib.pyplot as plt
# from diffusion_policy.env.pusht.pusht_env import PushTEnv
# env = PushTEnv(render_size=96)

dataset = get_dataset()
obs = []
policy = load_policy_from_checkpoint("models/pusht/0400-test_mean_score=0.844.ckpt")

for ts in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()

    state = observation["observation.state"] # 6 DoF
    image = observation["observation.images.phone"]

    # state -> x,y
    print("state: ", state)
    state = state.cpu().numpy()
    x, y = forward_kinematics(dataset, state)
    print("forward kinematics: ", x, y)

    # image -> x_T, y_T, theta_T
    # print("image", image.type())
    image = image.cpu().numpy()
    x_T, y_T, theta_T = get_t_position_and_orientation(image)
    x_T = x_T / 25.6
    y_T = y_T / 30
    theta_T = theta_T * np.pi / 180 # deg -> rad
    print("overlayed grid cartesian coords [0, 16]: ", x, y, x_T, y_T, theta_T)
    # grid coords -> sim coords
    x_sim, y_sim, x_T_sim, y_T_sim, theta_T_sim = coordinate_transform(x, y, x_T, y_T, theta_T)
    # pixel -> grid
    print("sim coords [0, 512]: ", x_sim, y_sim, x_T_sim, y_T_sim, theta_T_sim)

    # model takes in observation context window
    obs.append((x_sim, y_sim, x_T_sim, y_T_sim, theta_T_sim))
    np_obs = np.array(obs)
    if len(obs) < 2:
        np_obs = np.array([obs[0], obs[0]])
    # else:
    #     np_obs = np.array([obs[-2:]]) # TODO: in policy_step
    # assert np_obs.shape == (2,5), f"np_obs.shape: {np_obs.shape}"

    sim_action = policy_step(np_obs, policy, device)
    sim_action = sim_action[0] # get first of action trajectory
    print("sim action: ", sim_action)

    # x,y -> action
    action = inverse_kinematics(dataset, sim_action)
    action = torch.from_numpy(action)
    print("action: ", action)
    # Order the robot to move
    robot.send_action(action)

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    # for name in observation:
    #     if "image" in name:
    #         observation[name] = observation[name].type(torch.float32) / 255
    #         observation[name] = observation[name].permute(2, 0, 1).contiguous()
    #         frames.append(observation[name])
    #     observation[name] = observation[name].unsqueeze(0)
    #     observation[name] = observation[name].to(device)

    # # Compute the next action with the policy
    # # based on the current observation
    # s_time = time.perf_counter()
    # action = policy.select_action(observation)
    # e_time = time.perf_counter()
    # frame_time = e_time - s_time
    # print(f"Time taken: {e_time - s_time} seconds")
    # # Remove batch dimension
    # action = action.squeeze(0)
    # # Move to cpu, if not already the case
    # action = action.to("cpu")
    # # Order the robot to move
    # robot.send_action(action)

    # log_entry = {
    #     "frame": ts,
    #     "frame_time": frame_time,
    #     "observation_state": observation["observation.state"].cpu().numpy().tolist(),
    #     "action": action.cpu().numpy().tolist()
    # }
    # log_data.append(log_entry)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)

total_time = time.perf_counter() - s_total
print(f"Total time taken: {total_time} seconds")