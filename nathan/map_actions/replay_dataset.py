import gymnasium as gym
import gym_pusht
import numpy as np
import matplotlib.pyplot as plt
from nathan.loader import DatasetLoader
import time
import argparse
import pygame

def map_action(action, source_shape=(6,), target_shape=(2,), mapping_method="simple", scale_factor=1.0):
    """
    Map actions from source shape to target shape using different mapping methods.
    
    Args:
        action (np.ndarray): The original action
        source_shape (tuple): The expected shape of the source action
        target_shape (tuple): The target shape for the environment
        mapping_method (str): The mapping method to use
            - "simple": Just take the first n dimensions
            - "xy_only": Take only x and y components (first two dimensions)
            - "custom": Apply a custom transformation
            - "normalized": Normalize actions to [-1, 1] range
        scale_factor (float): Scale factor to apply to the actions
    
    Returns:
        np.ndarray: The mapped action with the target shape
    """
    if not isinstance(action, np.ndarray):
        action = np.array(action)
    
    # If shapes already match, return as is
    if action.shape == target_shape:
        return action * scale_factor
    
    # If action doesn't match expected source shape, print warning
    if action.shape != source_shape and source_shape is not None:
        print(f"Warning: Unexpected source action shape. Got {action.shape}, expected {source_shape}")
    
    if mapping_method == "simple":
        # Simply take the first n dimensions
        return action[:target_shape[0]] * scale_factor
    
    elif mapping_method == "xy_only":
        # Take only x and y components (first two dimensions)
        # For PushT, we need to ensure these are in the right range
        # The environment expects values in [-1, 1]
        xy_action = action[:2]
        
        # Check if values are already in [-1, 1] range
        if np.all(np.abs(xy_action) <= 1.0):
            return xy_action * scale_factor
        else:
            # Normalize to [-1, 1] range if they're not
            max_val = max(np.abs(xy_action).max(), 1e-10)  # Avoid division by zero
            return (xy_action / max_val) * scale_factor
    
    elif mapping_method == "normalized":
        # Normalize the action to [-1, 1] range
        action_subset = action[:target_shape[0]]
        max_val = max(np.abs(action_subset).max(), 1e-10)  # Avoid division by zero
        return (action_subset / max_val) * scale_factor
    
    elif mapping_method == "custom":
        # Example of a custom mapping - adjust as needed for your specific dataset
        if source_shape == (6,) and target_shape == (2,):
            # This is just an example - you might need a different mapping
            return np.array([
                action[0],  # x-component
                action[1],  # y-component
            ]) * scale_factor
    
    # Default fallback - simple slicing
    return action[:target_shape[0]] * scale_factor

def replay_episode(episode_id=0, render=True, save_video=False, dataset_name="ellen2imagine/pusht_green1", 
                  mapping_method="xy_only", scale_factor=1.0, skip_frames=1):
    """
    Replay a specific episode from the dataset in the PushT simulator.
    
    Args:
        episode_id (int): The episode ID to replay
        render (bool): Whether to render the environment
        save_video (bool): Whether to save a video of the replay
        dataset_name (str): The name of the dataset to use
        mapping_method (str): Method to map actions from dataset to environment
        scale_factor (float): Scale factor to apply to the actions
        skip_frames (int): Number of frames to skip between each action
    """
    # Load the dataset
    loader = DatasetLoader(dataset_name)
    
    # Get the episode data
    print(f"Loading episode {episode_id}...")
    episode_data = loader.get_episode_data(episode_id)
    
    if "parquet_data" not in episode_data:
        print(f"Error: Could not load parquet data for episode {episode_id}")
        return
    
    # Extract the action data
    df = episode_data["parquet_data"]
    
    # Check if 'action' column exists
    if 'action' not in df.columns:
        print(f"Error: 'action' column not found in parquet data. Available columns: {df.columns}")
        return
    
    actions = df['action'].tolist()
    print(f"Loaded {len(actions)} actions from episode {episode_id}")
    
    # Apply frame skipping
    if skip_frames > 1:
        actions = actions[::skip_frames]
        print(f"Applied frame skipping. Using {len(actions)} actions after skipping.")
    
    # Create the environment with appropriate render mode
    render_mode = "rgb_array" if save_video else "human" if render else None
    env = gym.make("gym_pusht/PushT-v0", render_mode=render_mode)
    observation, info = env.reset()
    
    # Setup for video recording if requested
    frames = []
    
    # Print action space information
    print(f"Environment action space: {env.action_space}")
    print(f"Action space low: {env.action_space.low}")
    print(f"Action space high: {env.action_space.high}")
    
    # Determine source shape from the first action
    source_shape = None
    if actions and len(actions) > 0:
        first_action = actions[0]
        if isinstance(first_action, np.ndarray):
            source_shape = first_action.shape
        else:
            try:
                source_shape = np.array(first_action).shape
            except:
                print("Warning: Could not determine source action shape")
    
    print(f"Source action shape: {source_shape}")
    print(f"Target action shape: {env.action_space.shape}")
    print(f"Using mapping method: {mapping_method}")
    print(f"Using scale factor: {scale_factor}")
    
    # Print first few actions for debugging
    print("\nFirst 5 actions from dataset:")
    for i, action in enumerate(actions[:5]):
        print(f"Original action {i}: {action}")
        mapped_action = map_action(
            action, 
            source_shape=source_shape, 
            target_shape=env.action_space.shape,
            mapping_method=mapping_method,
            scale_factor=scale_factor
        )
        print(f"Mapped action {i}: {mapped_action}")
    
    # Replay the actions
    print(f"\nReplaying {len(actions)} actions...")
    for i, action in enumerate(actions):
        try:
            # Process pygame events to prevent window from freezing
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
            
            # Map the action to the target shape
            mapped_action = map_action(
                action, 
                source_shape=source_shape, 
                target_shape=env.action_space.shape,
                mapping_method=mapping_method,
                scale_factor=scale_factor
            )
            
            # Ensure the action is within the action space bounds
            mapped_action = np.clip(mapped_action, env.action_space.low, env.action_space.high)
            
            # Execute the action
            observation, reward, terminated, truncated, info = env.step(mapped_action)
            
            # Render and save frame if requested
            if save_video:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            elif render:
                env.render()  # Let the environment handle rendering
            
            # Add a small delay to make the replay visible
            if render:
                time.sleep(0.01)  # Reduced delay to make it more responsive
            
            # Print progress
            if i % 100 == 0:
                print(f"Replayed {i}/{len(actions)} actions")
            
            if terminated or truncated:
                print(f"Episode ended early at step {i}/{len(actions)}")
                break
                
        except Exception as e:
            print(f"Error at step {i}: {e}")
            break
    
    # Save video if requested
    if save_video and frames:
        try:
            from matplotlib import animation
            
            # Create figure and axes
            fig, ax = plt.subplots()
            
            # Create animation
            def init():
                ax.clear()
                return []
            
            def animate(i):
                ax.clear()
                ax.imshow(frames[i])
                ax.set_title(f"Frame {i}")
                ax.axis('off')
                return []
            
            anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                          frames=len(frames), interval=50, blit=True)
            
            # Save animation
            output_path = f"episode_{episode_id}_replay.mp4"
            anim.save(output_path, writer='ffmpeg', fps=20)
            print(f"Saved video to {output_path}")
            
        except Exception as e:
            print(f"Error saving video: {e}")
    
    env.close()
    print("Replay complete")

def main():
    parser = argparse.ArgumentParser(description='Replay a dataset episode in the PushT simulator')
    parser.add_argument('--episode', type=int, default=0, help='Episode ID to replay')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--save-video', action='store_true', help='Save a video of the replay')
    parser.add_argument('--dataset', type=str, default="ellen2imagine/pusht_green1", 
                        help='Dataset name (default: ellen2imagine/pusht_green1)')
    parser.add_argument('--mapping', type=str, default="xy_only", 
                        choices=["simple", "xy_only", "custom", "normalized"],
                        help='Action mapping method (default: xy_only)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor to apply to actions (default: 1.0)')
    parser.add_argument('--skip', type=int, default=1,
                        help='Number of frames to skip between each action (default: 1)')
    
    args = parser.parse_args()
    
    replay_episode(
        episode_id=args.episode,
        render=not args.no_render,
        save_video=args.save_video,
        dataset_name=args.dataset,
        mapping_method=args.mapping,
        scale_factor=args.scale,
        skip_frames=args.skip
    )

if __name__ == "__main__":
    main() 