# runs open loop sim2real pipeline (no feedback from PushT action)
# we start the sim from the real state, then compute the optimal action
# only in sim, and updating the T location based on camera

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from lerobot.scripts.control_robot import busy_wait
from lerobot.scripts.configure_robot import configure_robot
from helper import *
from run_pusht import load_policy_from_checkpoint, policy_step
from diffusion_policy.env.pusht.pusht_env import PushTEnv

def main(env, policy, render_mode="human", inference_time=20, fps=10, device="cuda"):
    # get initial observation from real
    observation = robot.capture_observation()
    state = observation["observation.state"].cpu().numpy()
    image = observation["observation.images.phone"].cpu().numpy()

    x, y = forward_kinematics(kinematics_dataset, state, use_model=False)
    x_T, y_T, theta_T = get_position_T_from_image(image)
    x_sim, y_sim, x_T_sim, y_T_sim, theta_T_sim = real2sim_transform(x, y, x_T, y_T, theta_T)

    # put initial state in env
    obs = env.reset()
    initial_obs = (x_sim, y_sim, x_T_sim, y_T_sim, theta_T_sim)
    env.set_state(*initial_obs)
    # obs = np.array([initial_obs, initial_obs])

    # dffusion takes in 2 steps of context window, so we pad the initial state
    np_obs = np.array([initial_obs, initial_obs])
    frames = []
    # action_queue = []
    log_data = []
    for ts in range(inference_time * fps):
        start_time = time.perf_counter()

        action = policy_step(np_obs, policy, device)
        # get next action
        action = action[0]
        obs, reward, done, info = env.step(action)

        # # action chunking (execute 8 actions in sequence, speedup)
        # if len(action_queue) == 0:
        #   action = policy_step(np_obs, policy, device)
        #   action_queue.extend(action)
        # action = action_queue.pop(0)

        np_obs = np.array([obs, obs])

        # update the T location
        observation = robot.capture_observation()
        image = observation["observation.images.phone"].cpu().numpy()
        x_T, y_T, theta_T = get_position_T_from_image(image)
        
        # convert T position to sim coords
        _, _, x_T_sim, y_T_sim, theta_T_sim = real2sim_transform(x, y, x_T, y_T, theta_T)

        # x,y are still the same as in the obs
        # update the T location
        x, y = obs[0], obs[1]
        env.set_state(x, y, x_T_sim, y_T_sim, theta_T_sim)
        env.render(render_mode)

        #### SEND TO ROBOT ####

        # sim coords -> grid coords    
        action = sim2real_transform(*action)
        # print("overlayed grid action: ", action)

        # x,y -> action
        action = inverse_kinematics(kinematics_dataset, np.array(action))
        action = torch.from_numpy(action)
        # print("action: ", action)

        # Order the robot to move
        robot.send_action(action)
        log_entry = {
            "frame": ts,
            "frame_time": time.perf_counter() - start_time,
            "observation_state": observation["observation.state"].cpu().numpy().tolist(),
            "action": action.cpu().numpy().tolist()
        }
        log_data.append(log_entry)
        frames.append(image)

        dt_s = time.perf_counter() - start_time
        busy_wait(1 / fps - dt_s)

    return frames, log_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/pusht/0400-test_mean_score=0.844.ckpt")
    parser.add_argument("--render_mode", type=str, default="human")
    parser.add_argument("--render_action", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--inference_time", type=int, default=20) # seconds
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    robot = configure_robot()

    # set up env
    # env = MultiStepWrapper(
    #     PushTEnv(render_size=96),#, render_mode=render_mode),
    #     n_obs_steps=2,
    #     n_action_steps=8,
    #     max_episode_steps=inference_time_s * fps
    # )
    env = PushTEnv(render_size=96)
    env.seed(args.seed)
    # set up sim2real
    kinematics_dataset = get_dataset()
    policy = load_policy_from_checkpoint(args.checkpoint, device=args.device)

    s_total = time.perf_counter()
    frames, log_data = main(env, policy, args.render_mode, args.inference_time, args.fps, args.device)
    total_time = time.perf_counter() - s_total
    print(f"Total time taken: {total_time} seconds")
    
    i = 0 # change this for rollout id
    output_dir = "outputs/eval/diffusion_sim2real"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/rollout_{i}"
    
    from lerobot.scripts.save_results import save_results
    save_results(frames, log_data, output_filename, total_time, args.fps, args.inference_time, args.device)
    env.close()
