import numpy as np
import matplotlib.pyplot as plt
from loader import DatasetLoader
import argparse
import gymnasium as gym
import gym_pusht

def analyze_actions(episode_id=0, dataset_name="ellen2imagine/pusht_green1"):
    """
    Analyze the action format in the dataset and compare with the simulator's action space.
    
    Args:
        episode_id (int): The episode ID to analyze
        dataset_name (str): The name of the dataset to use
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
    
    # Analyze the first few actions
    print("\nFirst 5 actions:")
    for i, action in enumerate(actions[:5]):
        print(f"Action {i}: {action}")
        if hasattr(action, 'shape'):
            print(f"  Shape: {action.shape}")
        print(f"  Type: {type(action)}")
    
    # Create the environment to get action space info
    env = gym.make("gym_pusht/PushT-v0")
    print(f"\nEnvironment action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Action space bounds: [{env.action_space.low}, {env.action_space.high}]")
    env.close()
    
    # Analyze action statistics
    if len(actions) > 0:
        # Convert all actions to numpy arrays for analysis
        action_arrays = []
        for action in actions:
            if isinstance(action, np.ndarray):
                action_arrays.append(action)
            else:
                try:
                    action_arrays.append(np.array(action))
                except:
                    print(f"Warning: Could not convert action to numpy array: {action}")
        
        if action_arrays:
            # Stack all actions into a single array
            all_actions = np.stack(action_arrays)
            
            # Calculate statistics
            action_min = np.min(all_actions, axis=0)
            action_max = np.max(all_actions, axis=0)
            action_mean = np.mean(all_actions, axis=0)
            action_std = np.std(all_actions, axis=0)
            
            print("\nAction statistics:")
            print(f"Min: {action_min}")
            print(f"Max: {action_max}")
            print(f"Mean: {action_mean}")
            print(f"Std: {action_std}")
            
            # Plot action distributions
            fig, axes = plt.subplots(all_actions.shape[1], 1, figsize=(10, 3*all_actions.shape[1]))
            if all_actions.shape[1] == 1:
                axes = [axes]
                
            for i in range(all_actions.shape[1]):
                axes[i].hist(all_actions[:, i], bins=50)
                axes[i].set_title(f'Action dimension {i}')
                axes[i].axvline(action_mean[i], color='r', linestyle='dashed', linewidth=1)
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(f"action_distribution_episode_{episode_id}.png")
            print(f"Saved action distribution plot to action_distribution_episode_{episode_id}.png")
            
            # Plot action sequences over time
            fig, axes = plt.subplots(all_actions.shape[1], 1, figsize=(15, 3*all_actions.shape[1]))
            if all_actions.shape[1] == 1:
                axes = [axes]
                
            for i in range(all_actions.shape[1]):
                axes[i].plot(all_actions[:, i])
                axes[i].set_title(f'Action dimension {i} over time')
                axes[i].set_xlabel('Step')
                axes[i].set_ylabel('Value')
            
            plt.tight_layout()
            plt.savefig(f"action_sequence_episode_{episode_id}.png")
            print(f"Saved action sequence plot to action_sequence_episode_{episode_id}.png")

def main():
    parser = argparse.ArgumentParser(description='Analyze actions in a dataset episode')
    parser.add_argument('--episode', type=int, default=0, help='Episode ID to analyze')
    parser.add_argument('--dataset', type=str, default="ellen2imagine/pusht_green1", 
                        help='Dataset name (default: ellen2imagine/pusht_green1)')
    
    args = parser.parse_args()
    
    analyze_actions(
        episode_id=args.episode,
        dataset_name=args.dataset
    )

if __name__ == "__main__":
    main() 