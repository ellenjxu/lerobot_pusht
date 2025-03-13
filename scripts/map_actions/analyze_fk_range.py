import numpy as np
import matplotlib.pyplot as plt
from scripts.loader import DatasetLoader
from lerobot_kinematics import lerobot_FK, get_robot
import argparse
from tqdm import tqdm

def analyze_fk_range(dataset_name="ellen2imagine/pusht_green1", num_episodes=10, max_actions_per_episode=None):
    """
    Analyze the range of x, y, z positions from the LeRobot forward kinematics function
    for actions in the dataset.
    
    Args:
        dataset_name (str): The name of the dataset to analyze
        num_episodes (int): Number of episodes to analyze
        max_actions_per_episode (int): Maximum number of actions to analyze per episode
    """
    # Load the dataset
    loader = DatasetLoader(dataset_name)
    
    # Get the robot model
    robot = get_robot()
    
    # Lists to store x, y, z positions
    x_positions = []
    y_positions = []
    z_positions = []
    
    # Analyze episodes
    print(f"Analyzing {num_episodes} episodes from dataset {dataset_name}...")
    
    for episode_id in range(num_episodes):
        try:
            # Get the episode data
            episode_data = loader.get_episode_data(episode_id)
            
            if "parquet_data" not in episode_data:
                print(f"Warning: Could not load parquet data for episode {episode_id}")
                continue
            
            # Extract the action data
            df = episode_data["parquet_data"]
            
            # Check if 'action' column exists
            if 'action' not in df.columns:
                print(f"Warning: 'action' column not found in parquet data for episode {episode_id}")
                continue
            
            # Get actions
            actions = df['action'].tolist()
            
            # Limit the number of actions if specified
            if max_actions_per_episode is not None:
                actions = actions[:max_actions_per_episode]
            
            print(f"Processing {len(actions)} actions from episode {episode_id}...")
            
            # Process each action
            for action in tqdm(actions):
                try:
                    # Convert to numpy array if needed
                    if not isinstance(action, np.ndarray):
                        action = np.array(action)
                    
                    # Apply forward kinematics
                    x, y, z = lerobot_FK(action, robot=robot)
                    
                    # Store positions
                    x_positions.append(x)
                    y_positions.append(y)
                    z_positions.append(z)
                    
                except Exception as e:
                    print(f"Error processing action: {e}")
                    continue
            
        except Exception as e:
            print(f"Error processing episode {episode_id}: {e}")
            continue
    
    # Calculate statistics
    if not x_positions:
        print("No valid positions found.")
        return
    
    # Convert to numpy arrays for easier analysis
    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    z_positions = np.array(z_positions)
    
    # Calculate statistics
    stats = {
        'x': {
            'min': np.min(x_positions),
            'max': np.max(x_positions),
            'mean': np.mean(x_positions),
            'std': np.std(x_positions),
            'range': np.max(x_positions) - np.min(x_positions)
        },
        'y': {
            'min': np.min(y_positions),
            'max': np.max(y_positions),
            'mean': np.mean(y_positions),
            'std': np.std(y_positions),
            'range': np.max(y_positions) - np.min(y_positions)
        },
        'z': {
            'min': np.min(z_positions),
            'max': np.max(z_positions),
            'mean': np.mean(z_positions),
            'std': np.std(z_positions),
            'range': np.max(z_positions) - np.min(z_positions)
        }
    }
    
    # Print statistics
    print("\nPosition Statistics:")
    print("===================")
    
    for axis, axis_stats in stats.items():
        print(f"\n{axis.upper()} Axis:")
        print(f"  Min: {axis_stats['min']:.6f}")
        print(f"  Max: {axis_stats['max']:.6f}")
        print(f"  Mean: {axis_stats['mean']:.6f}")
        print(f"  Std Dev: {axis_stats['std']:.6f}")
        print(f"  Range: {axis_stats['range']:.6f}")
    
    # Create scatter plot of x-y positions
    plt.figure(figsize=(10, 8))
    plt.scatter(x_positions, y_positions, alpha=0.5, s=5)
    plt.title(f'X-Y End Effector Positions from {dataset_name}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    
    # Add statistics to the plot
    stats_text = (
        f"X: [{stats['x']['min']:.3f}, {stats['x']['max']:.3f}], Range: {stats['x']['range']:.3f}\n"
        f"Y: [{stats['y']['min']:.3f}, {stats['y']['max']:.3f}], Range: {stats['y']['range']:.3f}"
    )
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Save the plot
    plt.savefig(f"{dataset_name.replace('/', '_')}_xy_positions.png")
    print(f"Saved plot to {dataset_name.replace('/', '_')}_xy_positions.png")
    
    # Show the plot
    plt.show()
    
    # Create histogram of x, y, z positions
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    axes[0].hist(x_positions, bins=50, alpha=0.7)
    axes[0].set_title('X Position Distribution')
    axes[0].grid(True)
    
    axes[1].hist(y_positions, bins=50, alpha=0.7)
    axes[1].set_title('Y Position Distribution')
    axes[1].grid(True)
    
    axes[2].hist(z_positions, bins=50, alpha=0.7)
    axes[2].set_title('Z Position Distribution')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{dataset_name.replace('/', '_')}_position_histograms.png")
    print(f"Saved histograms to {dataset_name.replace('/', '_')}_position_histograms.png")
    
    # Show the histograms
    plt.show()
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Analyze FK range in a dataset')
    parser.add_argument('--dataset', type=str, default="ellen2imagine/pusht_green1", 
                        help='Dataset name (default: ellen2imagine/pusht_green1)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to analyze (default: 10)')
    parser.add_argument('--max-actions', type=int, default=None,
                        help='Maximum number of actions to analyze per episode (default: all)')
    
    args = parser.parse_args()
    
    analyze_fk_range(
        dataset_name=args.dataset,
        num_episodes=args.episodes,
        max_actions_per_episode=args.max_actions
    )

if __name__ == "__main__":
    main() 