# runs sim2real pipeline (closed loop)
# TODO: fix bug where the robot arm is not moving, forward kinematics

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from lerobot.scripts.control_robot import busy_wait
from lerobot.scripts.configure_robot import configure_robot
from helper import *
from run_pusht import load_policy_from_checkpoint, policy_step
# optional: start the env for render
from diffusion_policy.env.pusht.pusht_env_live import PushTEnvLive

def main(env, policy, render_mode="human", inference_time=20, fps=10, device="cuda"):
    env.reset()
    obs = []
    for ts in range(inference_time * fps):
        start_time = time.perf_counter()

        # get image, state from real
        observation = robot.capture_observation()
        state = observation["observation.state"].cpu().numpy()
        image = observation["observation.images.phone"].cpu().numpy()

        # state -> x,y
        # print("state: ", state)
        x, y = forward_kinematics(kinematics_dataset, state, use_model=False)
        # print("forward kinematics: ", x, y)

        # image -> x_T, y_T, theta_T
        # print("image", image.type())
        x_T, y_T, theta_T = get_position_T_from_image(image)
        # print("overlayed grid cartesian coords [0, 16]: ", x, y, x_T, y_T, theta_T)
        # grid coords -> sim coords
        x_sim, y_sim, x_T_sim, y_T_sim, theta_T_sim = real2sim_transform(x, y, x_T, y_T, theta_T)
        # pixel -> grid
        # print("sim coords [0, 512]: ", x_sim, y_sim, x_T_sim, y_T_sim, theta_T_sim)

        # render the state that was computed from image (where agent currently is)
        env.set_state(x_sim, y_sim, x_T_sim, y_T_sim, theta_T_sim)
        env.render(render_mode)

        # model takes in observation context window
        obs.append((x_sim, y_sim, x_T_sim, y_T_sim, theta_T_sim))
        np_obs = np.array(obs)
        if len(obs) < 2:
            np_obs = np.array([obs[0], obs[0]])
        else:
            np_obs = np_obs[-2:]
        assert np_obs.shape == (2,5), f"np_obs.shape: {np_obs.shape}"

        sim_action = policy_step(np_obs, policy, device) # agent (x,y)
        sim_action = sim_action[0]
        # print("sim pred: ", sim_action)

        # render the action predicted by diffusion (where agent wants to be)
        env.latest_action = sim_action
        env.render(render_mode)

        # sim coords -> grid coords    
        action = sim2real_transform(*sim_action)
        # print("overlayed grid action: ", action)

        # x,y -> action
        action = inverse_kinematics(kinematics_dataset, np.array(action))
        action = torch.from_numpy(action)
        # print("action: ", action)

        # Order the robot to move
        robot.send_action(action)

        dt_s = time.perf_counter() - start_time
        busy_wait(1 / fps - dt_s)

    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/pusht/0400-test_mean_score=0.844.ckpt")
    parser.add_argument("--render_mode", type=str, default="human")
    parser.add_argument("--render_action", type=bool, default=True) # set target action to render
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--inference_time", type=int, default=20) # seconds
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    robot = configure_robot()

    # set up sim2real
    kinematics_dataset = get_dataset()
    policy = load_policy_from_checkpoint(args.checkpoint, device=args.device)
    env = PushTEnvLive(render_size=96, render_action=args.render_action)

    s_total = time.perf_counter()
    main(env, policy, args.render_mode, args.inference_time, args.fps, args.device)
    total_time = time.perf_counter() - s_total
    print(f"Total time taken: {total_time} seconds")