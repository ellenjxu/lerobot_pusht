"""
Modified from pusht_runner.py and eval.py for loading a policy and rolling out a single episode.
"""

import os
import torch
import dill
import hydra
import numpy as np
import pathlib
from tqdm import tqdm
import imageio
from diffusion_policy.env.pusht.pusht_env import PushTEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.common.pytorch_util import dict_apply

def load_policy_from_checkpoint(checkpoint_path, device='cuda:0'):
    """Load policy from checkpoint file"""
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    
    # Create temporary output dir for workspace
    temp_output_dir = os.path.join(os.path.dirname(checkpoint_path), 'temp_eval')
    pathlib.Path(temp_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize workspace and load model
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get policy from workspace
    policy = workspace.model
    if hasattr(workspace, 'ema_model') and cfg.training.use_ema:
        policy = workspace.ema_model
    
    policy.to(device)
    policy.eval()
    
    return policy

def run_pusht_with_policy(policy, seed=0, max_steps=200, n_obs_steps=2, n_action_steps=8, n_latency_steps=0, render_mode="human", video_path=None):
    """Run the PushT environment with the given policy."""
    device = policy.device
    
    env = MultiStepWrapper(
        PushTEnv(render_size=96),#, render_mode=render_mode),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps
    )
    
    # start rollout
    env.seed(seed)
    obs = env.reset()
    policy.reset()
    env.render(mode=render_mode)

    done = False
    total_reward = 0
    past_action = None
    frames = []
    
    pbar = tqdm(total=max_steps, desc="Running PushT environment")
    
    while not done:
        # create obs dict
        np_obs_dict = {
            # handle n_latency_steps by discarding the last n_latency_steps
            'obs': obs[np.newaxis, :n_obs_steps, :].astype(np.float32)  # use np.newaxis since we're not using VectorEnv
        }
        
        # not implemented yet
        # if past_action is not None:
        #     np_obs_dict['past_action'] = past_action[:, -(n_obs_steps-1):].astype(np.float32)
        
        # device transfer
        obs_dict = dict_apply(np_obs_dict, 
            lambda x: torch.from_numpy(x).to(device=device))
        
        # run policy
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)
        
        # device transfer
        np_action_dict = dict_apply(action_dict,
            lambda x: x.detach().to('cpu').numpy())
        
        # handle latency_steps, we discard the first n_latency_steps actions
        # to simulate latency
        action = np_action_dict['action'][:,n_latency_steps:]
        
        # print(action.shape)
        action = action[0]
        
        # step env
        obs, reward, done, info = env.step(action)
        total_reward += reward
        past_action = action

        if render_mode == "rgb_array":
            frame = env.render(mode=render_mode)
            frames.append(frame)
        else:
            env.render(mode=render_mode)
        
        # update pbar
        pbar.update(action.shape[0])
    
    pbar.close()
    print(f"episode reward: {total_reward}")
    
    if render_mode == "rgb_array" and frames:
        imageio.mimsave(video_path, frames, fps=10)
        print(f"Video saved to {video_path}")
    
    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="models/pusht/0400-test_mean_score=0.844.ckpt", help='Path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--render_mode', type=str, default='human', help='Render mode (human or rgb_array)')
    parser.add_argument('--video_path', type=str, default=None, help='Path to save video')
    args = parser.parse_args()
    
    if args.video_path is None:
        if not os.path.exists("assets"):
            os.makedirs("assets")
        video_path = f"assets/pusht_rollout_seed_{args.seed}.mp4"
    else:
        video_path = args.video_path
    
    policy = load_policy_from_checkpoint(args.checkpoint, device=args.device)
    
    run_pusht_with_policy(policy, seed=args.seed, render_mode=args.render_mode, video_path=video_path)