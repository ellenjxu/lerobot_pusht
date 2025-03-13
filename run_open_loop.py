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

inference_time_s = 20 # 20s for rollout
fps = 10
device = "cuda"

s_total = time.perf_counter()
frames = []
log_data = []

import torch
from nathan.get_t_info.mask import get_t_position_and_orientation
from helper import *
from run_pusht import load_policy_from_checkpoint, policy_step
# optional: start the env for render
import matplotlib.pyplot as plt
from diffusion_policy.env.pusht.pusht_env import PushTEnv

dataset = get_dataset()
obs = []
# policy = load_policy_from_checkpoint("models/pusht/0400-test_mean_score=0.844.ckpt")
policy = load_policy_from_checkpoint("models/pusht/0500-test_mean_score=0.883.ckpt")

# get initial observation from real
observation = robot.capture_observation()
state = observation["observation.state"]
image = observation["observation.images.phone"]

print("Initial state: ", state)
state = state.cpu().numpy()
x, y = forward_kinematics(dataset, state, use_model=False)
print("Initial forward kinematics: ", x, y)

image = image.cpu().numpy()
x_T, y_T, theta_T = get_t_position_and_orientation(image)
x_T = x_T / 25.6
y_T = y_T / 30
theta_T = theta_T * np.pi / 180 # deg -> rad
print("Initial overlayed grid cartesian coords [0, 16]: ", x, y, x_T, y_T, theta_T)

x_sim, y_sim, x_T_sim, y_T_sim, theta_T_sim = real2sim_transform(x, y, x_T, y_T, theta_T)
print("Initial sim coords [0, 512]: ", x_sim, y_sim, x_T_sim, y_T_sim, theta_T_sim)

# set up env
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
# env = MultiStepWrapper(
#     PushTEnv(render_size=96),#, render_mode=render_mode),
#     n_obs_steps=2,
#     n_action_steps=8,
#     max_episode_steps=inference_time_s * fps
# )
env = PushTEnv(render_size=96)

env.seed(42)
obs = env.reset()
initial_obs = (x_sim, y_sim, x_T_sim, y_T_sim, theta_T_sim)
env.set_state(*initial_obs)
# obs = np.array([initial_obs, initial_obs])

np_obs = np.array([initial_obs, initial_obs])
for ts in range(inference_time_s * fps):
    start_time = time.perf_counter()

    action = policy_step(np_obs, policy, device)
    # TODO: do batch of 8 actions
    action = action[0]
    obs, reward, done, info = env.step(action)
    
    np_obs = np.array([obs, obs])
    # np_obs = np_obs[-2:]

    # update the T location
    observation = robot.capture_observation()
    image = observation["observation.images.phone"].cpu().numpy()
    x_T, y_T, theta_T = get_t_position_and_orientation(image)
    x_T = x_T / 25.6
    y_T = y_T / 30
    theta_T = theta_T * np.pi / 180 # deg -> rad
    
    # convert T position
    _, _, x_T_sim, y_T_sim, theta_T_sim = real2sim_transform(x, y, x_T, y_T, theta_T)

    # x,y are still the same as in the obs
    # update the T location
    x, y = obs[0], obs[1]
    env.set_state(x, y, x_T_sim, y_T_sim, theta_T_sim)
    env.render(mode="human")

    #### SEND TO ROBOT ####

    # sim coords -> grid coords    
    action = sim2real_transform(*action)
    print("overlayed grid action: ", action)

    # x,y -> action
    action = inverse_kinematics(dataset, np.array(action))
    action = torch.from_numpy(action)
    print("action: ", action)

    # Order the robot to move
    robot.send_action(action)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)
