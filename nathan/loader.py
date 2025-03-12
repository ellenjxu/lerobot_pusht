import os
import json
import pandas as pd
import requests
from huggingface_hub import hf_hub_download
import cv2

class DatasetLoader:
    def __init__(self, dataset_name="ellen2imagine/pusht_green1"):
        """
        Initialize the loader with the dataset name.
        
        Args:
            dataset_name (str): The Hugging Face dataset name (e.g., 'ellen2imagine/pusht_green1')
        """
        self.dataset_name = dataset_name
        self.cache_dir = os.path.join(os.getcwd(), "cache", dataset_name.split('/')[-1])
        self.metadata = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load the metadata JSON files if they exist in cache, otherwise download them."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # List of metadata files to download
        meta_files = ["episodes.jsonl", "info.json", "stats.json", "tasks.jsonl"]
        
        for filename in meta_files:
            try:
                file_path = hf_hub_download(
                    repo_id=self.dataset_name,
                    filename=f"meta/{filename}",
                    repo_type="dataset",
                    cache_dir=self.cache_dir
                )
                
                if filename.endswith('.jsonl'):
                    # For JSONL files, load each line as a separate JSON object
                    with open(file_path, 'r') as f:
                        self.metadata[filename] = [json.loads(line) for line in f.readlines() if line.strip()]
                else:
                    # For regular JSON files
                    with open(file_path, 'r') as f:
                        self.metadata[filename] = json.load(f)
                        
                print(f"Loaded {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    def get_episode_data(self, episode_id):
        """
        Get all data associated with a specific episode.
        
        Args:
            episode_id (int): The episode ID (e.g., 0, 1, 2)
            
        Returns:
            dict: A dictionary containing all data associated with the episode
        """
        # Format episode_id with leading zeros (6 digits)
        episode_id_str = f"{episode_id:06d}"
        
        result = {
            "id": episode_id,
            "metadata": self._get_episode_metadata(episode_id),
            "parquet_path": None,
            "video_path": None
        }
        
        # Download parquet file
        try:
            parquet_path = hf_hub_download(
                repo_id=self.dataset_name,
                filename=f"data/chunk-000/episode_{episode_id_str}.parquet",
                repo_type="dataset",
                cache_dir=self.cache_dir
            )
            result["parquet_path"] = parquet_path
            result["parquet_data"] = pd.read_parquet(parquet_path)
            print(f"Loaded parquet data for episode {episode_id_str}")
        except Exception as e:
            print(f"Error loading parquet data for episode {episode_id_str}: {e}")
        
        # Download video file
        try:
            video_path = hf_hub_download(
                repo_id=self.dataset_name,
                filename=f"videos/chunk-000/observation.images.phone/episode_{episode_id_str}.mp4",
                repo_type="dataset",
                cache_dir=self.cache_dir
            )
            result["video_path"] = video_path
            print(f"Loaded video for episode {episode_id_str}")
        except Exception as e:
            print(f"Error loading video for episode {episode_id_str}: {e}")
        
        return result
    
    def _get_episode_metadata(self, episode_id):
        """Extract metadata for a specific episode from the loaded metadata files."""
        result = {}
        
        # Extract episode info from episodes.jsonl if available
        if "episodes.jsonl" in self.metadata:
            for episode in self.metadata["episodes.jsonl"]:
                if episode.get("episode_id") == episode_id:
                    result["episode_info"] = episode
                    break
        
        # Add any other relevant metadata
        if "info.json" in self.metadata:
            result["dataset_info"] = self.metadata["info.json"]
        
        if "stats.json" in self.metadata:
            result["stats"] = self.metadata["stats.json"]
            
        return result
    
    def list_available_episodes(self):
        """List all available episodes in the dataset."""
        if "episodes.jsonl" in self.metadata:
            episodes = self.metadata["episodes.jsonl"]
            return [ep.get("episode_id") for ep in episodes if "episode_id" in ep]
        
        # If episodes.jsonl doesn't contain the info, try to infer from other sources
        try:
            # This will make an API call to get file listing from Hugging Face
            api_url = f"https://huggingface.co/api/datasets/{self.dataset_name}/tree/main/data/chunk-000"
            response = requests.get(api_url)
            if response.status_code == 200:
                files = response.json()
                episode_ids = []
                for file in files:
                    if file.get("type") == "file" and file.get("path").startswith("data/chunk-000/episode_") and file.get("path").endswith(".parquet"):
                        # Extract episode ID from filename
                        filename = os.path.basename(file.get("path"))
                        episode_id_str = filename.replace("episode_", "").replace(".parquet", "")
                        try:
                            episode_ids.append(int(episode_id_str))
                        except ValueError:
                            pass
                return sorted(episode_ids)
        except Exception as e:
            print(f"Error listing episodes: {e}")
        
        return []
    
    def get_dataset_info(self):
        """Get general information about the dataset."""
        result = {}
        
        for key, value in self.metadata.items():
            if key != "episodes.jsonl":  # This would make the output too verbose
                result[key] = value
        
        # Add a count of episodes
        result["episode_count"] = len(self.list_available_episodes())
        
        return result

    def get_frame_sequence(self, episode_id, start_frame=0, num_frames=10, skip_frames=5):
        """
        Get a sequence of frames from a specific episode with customizable frame skipping.
        
        Args:
            episode_id (int): The episode ID
            start_frame (int): The starting frame index
            num_frames (int): Number of frames to extract
            skip_frames (int): Number of frames to skip between each extracted frame
            
        Returns:
            dict: A dictionary containing the sequence information and frames
        """
        # Get episode data
        episode_data = self.get_episode_data(episode_id)
        
        if not episode_data or 'video_path' not in episode_data or not episode_data['video_path']:
            print(f"Error: Could not load video for episode {episode_id}")
            return None
        
        # Open the video file
        cap = cv2.VideoCapture(episode_data['video_path'])
        
        if not cap.isOpened():
            print(f"Error: Could not open video for episode {episode_id}")
            cap.release()
            return None
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract
        frame_indices = []
        for i in range(num_frames):
            frame_idx = start_frame + (i * skip_frames)
            if frame_idx < total_frames:
                frame_indices.append(frame_idx)
        
        # Extract frames
        frames = []
        frame_info = []
        
        for idx in frame_indices:
            # Set position to the requested frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            
            # Read the frame
            ret, frame = cap.read()
            
            if ret:
                # Convert from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_info.append({
                    'dataset': self.dataset_name,
                    'episode_id': episode_id,
                    'frame_index': idx
                })
            else:
                print(f"Warning: Could not read frame {idx}")
        
        # Release the video capture object
        cap.release()
        
        return {
            'episode_id': episode_id,
            'frames': frames,
            'frame_info': frame_info,
            'metadata': episode_data['metadata']
        }


# Example usage
if __name__ == "__main__":
    # Replace with your dataset name if different
    DATASET_NAME = "ellen2imagine/pusht_green1"
    
    loader = DatasetLoader(DATASET_NAME)
    
    # List available episodes
    episodes = loader.list_available_episodes()
    print(f"Available episodes: {episodes}")
    
    # Get dataset info
    info = loader.get_dataset_info()
    print(f"Dataset info: {info}")
    
    # Get data for episode 0
    episode_data = loader.get_episode_data(0)
    
    # Print parquet data summary if available
    if "parquet_data" in episode_data:
        print(f"Parquet data shape: {episode_data['parquet_data'].shape}")
        print(f"Parquet data columns: {episode_data['parquet_data'].columns}")
        print(f"First 5 rows:\n{episode_data['parquet_data'].head()}")
    
    # Print paths to files
    print(f"Parquet file path: {episode_data['parquet_path']}")
    print(f"Video file path: {episode_data['video_path']}")