import time
import cv2
from lerobot.scripts.control_robot import busy_wait
import os
import sys
import numpy as np
from lerobot.scripts.configure_robot import configure_robot

inference_time_s = 20 # 20s for rollout
fps = 10
device = "cuda"

s_total = time.perf_counter()
frames = []
log_data = []

import torch
from scripts.get_t_info.mask import get_t_position_and_orientation
from helper import *
from run_pusht import load_policy_from_checkpoint, policy_step
# optional: start the env for render
import matplotlib.pyplot as plt
from diffusion_policy.env.pusht.pusht_env_live import PushTEnvLive
env = PushTEnvLive(render_size=96, render_action=True)
env.reset()

dataset = get_dataset()
obs = []
policy = load_policy_from_checkpoint("models/pusht/0400-test_mean_score=0.844.ckpt")

for ts in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()

    state = observation["observation.state"].cpu().numpy()
    image = observation["observation.images.phone"].cpu().numpy()

    # state -> x,y
    print("state: ", state)
    x, y = forward_kinematics(dataset, state, use_model=False)
    print("forward kinematics: ", x, y)

    # image -> x_T, y_T, theta_T
    # print("image", image.type())
    x_T, y_T, theta_T = get_t_position_and_orientation(image)
    x_T = x_T / 25.6
    y_T = y_T / 30
    theta_T = theta_T * np.pi / 180 # deg -> rad
    print("overlayed grid cartesian coords [0, 16]: ", x, y, x_T, y_T, theta_T)
    # grid coords -> sim coords
    x_sim, y_sim, x_T_sim, y_T_sim, theta_T_sim = real2sim_transform(x, y, x_T, y_T, theta_T)
    # pixel -> grid
    print("sim coords [0, 512]: ", x_sim, y_sim, x_T_sim, y_T_sim, theta_T_sim)

    # render the state that was computed from image (where guy currently is)
    env.set_state(x_sim, y_sim, x_T_sim, y_T_sim, theta_T_sim)
    env.render("human")

    # model takes in observation context window
    obs.append((x_sim, y_sim, x_T_sim, y_T_sim, theta_T_sim))
    np_obs = np.array(obs)
    if len(obs) < 2:
        np_obs = np.array([obs[0], obs[0]])
    else:
        np_obs = np_obs[-2:]
    assert np_obs.shape == (2,5), f"np_obs.shape: {np_obs.shape}"

    sim_action = policy_step(np_obs, policy, device) # agent (x,y)
    sim_action = sim_action[0] # TODO: check get last of action trajectory
    print("sim pred: ", sim_action)

    # render the action predicted by diffusion (where guy wants to be)
    env.latest_action = sim_action
    env.render("human")

    # sim coords -> grid coords    
    action = sim2real_transform(*sim_action)
    print("overlayed grid action: ", action)

    # x,y -> action
    action = inverse_kinematics(dataset, np.array(action))
    action = torch.from_numpy(action)
    print("action: ", action)

    # Order the robot to move
    robot.send_action(action)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)

total_time = time.perf_counter() - s_total
print(f"Total time taken: {total_time} seconds")