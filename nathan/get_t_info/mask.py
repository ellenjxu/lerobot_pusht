if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
from skimage.measure import regionprops, label
from skimage.morphology import binary_closing, binary_opening, disk
from nathan.loader import DatasetLoader
import scipy.optimize as optimize

def extract_frame_from_video(video_path, frame_index=0):
    """
    Extract a specific frame from a video file.
    
    Args:
        video_path (str): Path to the video file
        frame_index (int): Index of the frame to extract (0-based)
        
    Returns:
        numpy.ndarray: The extracted frame as an RGB image, or None if extraction failed
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None
    
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_index >= total_frames:
            print(f"Error: Requested frame {frame_index} exceeds total frames {total_frames}")
            cap.release()
            return None
        
        # Set position to the requested frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        # Read the frame
        ret, frame = cap.read()
        
        # Release the video capture object
        cap.release()
        
        if not ret:
            print(f"Error: Could not read frame {frame_index}")
            return None
        
        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame_rgb
    
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return None

def get_t_mask(img, hex_color='#1f764f', tolerance=60):
    """
    Create a mask for the T-shaped object based on color.
    
    Args:
        img: Input RGB image
        hex_color: Target color in hex format (default: '#4CAF50' - a brighter green)
        tolerance: Color tolerance value
        
    Returns:
        Binary mask where the T-shaped object is marked as True
    """
    # Convert hex to RGB
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Create a single pixel array with the target color
    rgb_color = np.uint8([[[r, g, b]]])  # RGB order for consistency
    
    # Convert RGB to HSV
    hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
    h, s, v = int(hsv_color[0][0][0]), int(hsv_color[0][0][1]), int(hsv_color[0][0][2])
    
    # Create ranges with tolerance (ensuring all values are integers)
    h_lower = max(0, h - tolerance)
    h_upper = min(179, h + tolerance)
    s_lower = max(0, s - tolerance)
    s_upper = min(255, s + tolerance)  # Changed from s-tolerance to s+tolerance
    v_lower = max(0, v - tolerance)
    v_upper = min(255, v + tolerance)
    
    # Convert image to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Create mask
    mask = (h_lower <= hsv_img[:, :, 0]) & (hsv_img[:, :, 0] <= h_upper)
    mask &= (s_lower <= hsv_img[:, :, 1]) & (hsv_img[:, :, 1] <= s_upper)
    mask &= (v_lower <= hsv_img[:, :, 2]) & (hsv_img[:, :, 2] <= v_upper)
    
    return mask

def clean_mask(mask, closing_radius=5, opening_radius=3):
    """Apply morphological operations to clean the mask"""
    # Apply closing to fill small holes
    mask_closed = binary_closing(mask, disk(closing_radius))
    # Apply opening to remove small objects
    mask_cleaned = binary_opening(mask_closed, disk(opening_radius))
    return mask_cleaned

def get_t_properties(mask):
    """Extract properties of the T-shaped object from the mask"""
    # Label connected components in the mask
    labeled_mask = label(mask)
    
    # If no objects found, return None
    if np.max(labeled_mask) == 0:
        return None
    
    # Get properties of the largest object (assuming it's the T)
    regions = regionprops(labeled_mask)
    # Sort regions by area (descending) and take the largest one
    regions.sort(key=lambda x: x.area, reverse=True)
    t_region = regions[0]
    
    # Extract properties
    center_y, center_x = t_region.centroid
    orientation = t_region.orientation
    
    # Convert orientation from radians to degrees
    # Note: skimage's orientation is between -pi/2 and pi/2
    angle_deg = np.degrees(orientation)
    
    # Get the bounding box
    min_row, min_col, max_row, max_col = t_region.bbox
    height = max_row - min_row
    width = max_col - min_col
    
    # Get coordinates of all pixels in the region
    coords = np.column_stack(np.where(labeled_mask == t_region.label))
    
    # Calculate distances from center to each point
    dists = np.sqrt((coords[:, 0] - center_y)**2 + (coords[:, 1] - center_x)**2)
    max_dist = np.max(dists)
    
    # For T shape, we need to determine which way it's pointing
    # First, normalize the angle to 0-180 range
    if angle_deg < 0:
        angle_deg += 180
    
    # The T shape has a specific structure - a horizontal bar with a vertical stem
    # We need to analyze the distribution of pixels to determine the correct orientation
    
    # Try all four possible orientations (0, 90, 180, 270 degrees)
    # and see which one best matches a T shape
    best_score = -1
    best_angle = angle_deg
    
    for angle_offset in [0, 90, 180, 270]:
        test_angle = (angle_deg + angle_offset) % 360
        test_angle_rad = np.radians(test_angle)
        
        # Create rotation matrix
        rot_matrix = np.array([
            [np.cos(test_angle_rad), -np.sin(test_angle_rad)],
            [np.sin(test_angle_rad), np.cos(test_angle_rad)]
        ])
        
        # Center and rotate coordinates
        centered_coords = coords - np.array([center_y, center_x])
        rotated_coords = np.dot(centered_coords, rot_matrix.T)
        
        # For a T shape pointing "up", we expect:
        # 1. More points in the horizontal bar at the top
        # 2. A vertical stem extending downward
        
        # Define regions for the T shape
        # Horizontal bar region (top)
        h_bar_mask = (rotated_coords[:, 0] < -10) & (np.abs(rotated_coords[:, 1]) < 60)
        
        # Vertical stem region (bottom)
        v_stem_mask = (rotated_coords[:, 0] > 10) & (np.abs(rotated_coords[:, 1]) < 15)
        
        # Count points in each region
        h_bar_count = np.sum(h_bar_mask)
        v_stem_count = np.sum(v_stem_mask)
        
        # Calculate a score for this orientation
        # A good T shape should have points in both regions
        score = h_bar_count * v_stem_count
        
        if score > best_score:
            best_score = score
            best_angle = test_angle
    
    # Use the best angle
    angle_deg = best_angle
    
    # IMPORTANT FIX: Ensure the T is pointing in the correct direction
    # We need to check if we need to flip the T by 180 degrees
    # This is because the T shape is symmetric when rotated by 180 degrees
    
    # Rotate coordinates to the best orientation
    best_angle_rad = np.radians(angle_deg)
    rot_matrix = np.array([
        [np.cos(best_angle_rad), -np.sin(best_angle_rad)],
        [np.sin(best_angle_rad), np.cos(best_angle_rad)]
    ])
    
    centered_coords = coords - np.array([center_y, center_x])
    rotated_coords = np.dot(centered_coords, rot_matrix.T)
    
    # Check the distribution of points along the y-axis after rotation
    # For a T pointing "up", there should be more points in the bottom half
    top_half_count = np.sum(rotated_coords[:, 0] < 0)
    bottom_half_count = np.sum(rotated_coords[:, 0] >= 0)
    
    # If there are more points in the top half, flip the T by 180 degrees
    if top_half_count > bottom_half_count:
        angle_deg = (angle_deg + 180) % 360
    
    # Additional properties that might be useful
    properties = {
        'center_x': center_x,  # Using original center (the red dot from prior implementation)
        'center_y': center_y,  # Using original center (the red dot from prior implementation)
        'angle_deg': angle_deg,
        'area': t_region.area,
        'bbox': t_region.bbox,  # (min_row, min_col, max_row, max_col)
        'major_axis_length': t_region.major_axis_length,
        'minor_axis_length': t_region.minor_axis_length,
        'max_distance': max_dist,  # Maximum distance from center to any point
    }
    
    return properties

def create_t_shape(center_x, center_y, angle_deg, size=1.0):
    """
    Create a T shape with the given center, orientation and size.
    
    Args:
        center_x, center_y: Center coordinates of the T
        angle_deg: Orientation angle in degrees
        size: Scaling factor for the T size
        
    Returns:
        Array of points defining the T shape
    """
    # T dimensions (scaled by size)
    # These match the relative dimensions provided: 120x30 horizontal bar, 30x90 vertical stem
    # Adjusted to make the T shape more accurate
    h_width = 60 * size  # Half-width of horizontal bar
    h_height = 15 * size  # Half-height of horizontal bar
    v_width = 15 * size   # Half-width of vertical stem
    v_height = 60 * size  # Half-height of vertical stem (increased to match observed T)
    
    # Define T shape points (centered at junction point, pointing down)
    # IMPORTANT FIX: Changed the T shape to point down by default
    t_points = np.array([
        # Horizontal bar
        [-h_width, -h_height],
        [h_width, -h_height],
        [h_width, h_height],
        [-h_width, h_height],
        # Vertical stem (now pointing down)
        [-v_width, h_height],
        [-v_width, h_height + 2*v_height],
        [v_width, h_height + 2*v_height],
        [v_width, h_height]
    ])
    
    # Rotate points
    angle_rad = np.radians(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    rotated_points = np.dot(t_points, rot_matrix.T)
    
    # Translate to center
    t_shape = rotated_points + np.array([center_y, center_x])
    
    return t_shape

def calculate_overlap_score(mask, center_x, center_y, angle_deg, size):
    """
    Calculate how well a T shape overlaps with the given mask.
    
    Args:
        mask: Binary mask of the detected T
        center_x, center_y: Center coordinates to test
        angle_deg: Orientation angle in degrees to test
        size: Scale factor for the T size
        
    Returns:
        float: Overlap score (higher is better)
    """
    # Create a model T with the given parameters
    t_shape = create_t_shape(center_x, center_y, angle_deg, size)
    
    # Convert to integer points and create a mask for the model T
    t_points = t_shape[:, [1, 0]].astype(np.int32)  # Swap x,y for OpenCV
    t_points = t_points.reshape((-1, 1, 2))
    
    # Create mask for the model T
    h, w = mask.shape
    model_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Check if points are within image bounds
    if np.any(t_points[:, 0, 0] < 0) or np.any(t_points[:, 0, 0] >= w) or \
       np.any(t_points[:, 0, 1] < 0) or np.any(t_points[:, 0, 1] >= h):
        # Some points are outside the image, which would cause fillPoly to fail
        # Return a low score
        return 0.0
    
    cv2.fillPoly(model_mask, [t_points], 1)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask, model_mask).sum()
    model_area = model_mask.sum()
    mask_area = mask.sum()
    
    # Focus more on intersection than union for partial occlusions
    # This rewards models that cover as much of the visible mask as possible
    # while allowing for parts of the model to be outside the mask
    # (which would be expected in cases of occlusion)
    intersection_ratio = intersection / mask_area if mask_area > 0 else 0
    
    # Penalize excessive model area outside the mask, but not too harshly
    if model_area > 0:
        precision = intersection / model_area
    else:
        precision = 0
    
    # Combined score with more weight on intersection_ratio
    score = (0.7 * intersection_ratio) + (0.3 * precision)
    
    return score

def optimize_t_parameters_grid_search(mask, initial_props=None):
    """
    Find optimal T parameters using grid search to maximize overlap with the mask.
    
    Args:
        mask: Binary mask of the detected T
        initial_props: Optional initial properties as starting point
        
    Returns:
        dict: Optimized T properties
    """
    # Get region properties for initial guess and bounds
    labeled_mask = label(mask)
    if np.max(labeled_mask) == 0:
        return None
        
    regions = regionprops(labeled_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    t_region = regions[0]
    
    # Initial guess
    if initial_props is not None:
        initial_center_y = initial_props['center_y']
        initial_center_x = initial_props['center_x']
        initial_angle = initial_props['angle_deg']
    else:
        initial_center_y, initial_center_x = t_region.centroid
        initial_angle = np.degrees(t_region.orientation)
        if initial_angle < 0:
            initial_angle += 180
    
    # Estimate size based on region dimensions
    min_row, min_col, max_row, max_col = t_region.bbox
    width = max_col - min_col
    height = max_row - min_row
    initial_size = max(width, height) / 180  # Same scale factor as in visualization
    
    # Define search ranges
    # For position, search within a larger radius around the initial center
    search_radius = max(width, height) * 0.5  # Increased from 0.3 to 0.5 (50% of the object size)
    
    # Position offsets to try (in pixels) - increased number of points
    position_offsets = np.linspace(-search_radius, search_radius, 9)  # Increased from 7 to 9 points
    
    # Angles to try (in degrees) - increased range and resolution
    angle_range = 45  # Increased from 30 to 45 degrees
    angles = np.linspace(initial_angle - angle_range, initial_angle + angle_range, 11)  # Increased from 9 to 11 points
    
    # Also try the full 90-degree rotations to catch cases where orientation is completely off
    additional_angles = [(initial_angle + rot) % 360 for rot in [0, 90, 180, 270]]
    angles = np.unique(np.append(angles, additional_angles))
    
    # Sizes to try - increased range and resolution
    sizes = np.linspace(0.6 * initial_size, 1.4 * initial_size, 7)  # Wider range and more points
    
    # Find best parameters
    best_score = -1
    best_params = (initial_center_y, initial_center_x, initial_angle, initial_size)
    
    # First do a coarse search with the full 90-degree rotations
    for angle in additional_angles:
        score = calculate_overlap_score(mask, initial_center_x, initial_center_y, angle, initial_size)
        if score > best_score:
            best_score = score
            best_params = (initial_center_y, initial_center_x, angle, initial_size)
    
    # Use the best angle from coarse search as the new initial angle
    initial_angle = best_params[2]
    angles = np.linspace(initial_angle - angle_range, initial_angle + angle_range, 11)
    angles = np.unique(angles % 360)  # Normalize to 0-360
    
    # Now do the full grid search
    for y_offset in position_offsets:
        for x_offset in position_offsets:
            for angle in angles:
                for size in sizes:
                    # Test these parameters
                    center_y = initial_center_y + y_offset
                    center_x = initial_center_x + x_offset
                    
                    score = calculate_overlap_score(mask, center_x, center_y, angle, size)
                    
                    if score > best_score:
                        best_score = score
                        best_params = (center_y, center_x, angle, size)
    
    # Unpack best parameters
    best_center_y, best_center_x, best_angle, best_size = best_params
    
    # Create final properties dictionary
    optimized_props = {
        'center_x': best_center_x,
        'center_y': best_center_y,
        'angle_deg': best_angle,
        'area': t_region.area,
        'bbox': t_region.bbox,
        'major_axis_length': t_region.major_axis_length,
        'minor_axis_length': t_region.minor_axis_length,
        'max_distance': t_region.major_axis_length * 0.5,
        'size': best_size
    }
    
    # Add score for debugging/evaluation
    optimized_props['overlap_score'] = best_score
    
    return optimized_props

def visualize_t_detection(img, mask, properties=None):
    """
    Create a visualization of the T detection results.
    
    Args:
        img: Original RGB image
        mask: Binary mask of the detected T
        properties: Dictionary of T properties (if None, only the mask is visualized)
        
    Returns:
        numpy.ndarray: Visualization image
    """
    # Create a copy of the image for visualization
    vis_img = img.copy()
    
    # Create a colored mask overlay
    mask_overlay = np.zeros_like(vis_img)
    mask_overlay[mask > 0] = [0, 255, 0]  # Green for the mask
    
    # Add mask overlay with transparency
    vis_img = cv2.addWeighted(vis_img, 1.0, mask_overlay, 0.3, 0)
    
    # If properties are available, visualize them
    if properties is not None:
        # Extract center coordinates
        center_x = int(properties['center_x'])
        center_y = int(properties['center_y'])
        
        # Draw center point
        cv2.circle(vis_img, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot
        
        # Get bounding box for size estimation
        min_row, min_col, max_row, max_col = properties['bbox']
        width = max_col - min_col
        height = max_row - min_row
        
        # Determine T size based on properties
        if 'size' in properties:
            t_size = properties['size']
        else:
            # Estimate size from bounding box if not provided
            t_size = max(width, height) / 180  # Adjusted scale factor
        
        # Create and draw T shape overlay using the optimized parameters
        t_shape = create_t_shape(center_x, center_y, properties['angle_deg'], t_size)
        
        # Convert to integer points and reshape for cv2.polylines
        t_points = t_shape[:, [1, 0]].astype(np.int32)  # Swap x,y for OpenCV
        t_points = t_points.reshape((-1, 1, 2))
        
        # Draw T shape
        cv2.polylines(vis_img, [t_points], True, (255, 0, 255), 2)
        
        # Fill T shape with semi-transparent color
        t_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.fillPoly(t_mask, [t_points], 255)
        t_overlay = np.zeros_like(vis_img)
        t_overlay[t_mask > 0] = [255, 0, 255]  # Magenta fill
        vis_img = cv2.addWeighted(vis_img, 1.0, t_overlay, 0.3, 0)
        
        # Draw orientation line along the stem of the T
        # The T shape is defined with the stem pointing down when angle is 0
        angle_deg = properties['angle_deg']
        
        # Calculate the stem direction based on how create_t_shape works
        # When angle is 0, stem points down (270 degrees in standard coordinates)
        # So we need to add 270 to the angle to get the stem direction
        stem_angle_deg = (-angle_deg + 360) % 360  # or (angle_deg + 180) % 360
        stem_angle_rad = np.radians(stem_angle_deg)
        
        length = 100  # Length of the orientation line
        
        # Calculate end point of the orientation line
        end_x = int(center_x + length * np.cos(stem_angle_rad))
        end_y = int(center_y + length * np.sin(stem_angle_rad))
        
        # Draw the orientation line
        cv2.line(vis_img, (center_x, center_y), (end_x, end_y), (255, 0, 0), 3)  # Blue line
        
        # Draw bounding box
        min_row, min_col, max_row, max_col = properties['bbox']
        cv2.rectangle(vis_img, (min_col, min_row), (max_col, max_row), (255, 255, 0), 3)  # Thicker rectangle
        
        # Add text with properties - larger and with background
        font_scale = 1.0
        thickness = 2
        
        # Add background to text for better visibility
        text = f"Center: ({center_x:.1f}, {center_y:.1f})"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(vis_img, (10, 10), (10 + text_width, 10 + text_height + 10), (0, 0, 0), -1)
        cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        text = f"Angle: {properties['angle_deg']:.1f} deg"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(vis_img, (10, 40), (10 + text_width, 40 + text_height + 10), (0, 0, 0), -1)
        cv2.putText(vis_img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Add size information
        if 'size' in properties:
            text = f"Size: {properties['size']:.3f}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(vis_img, (10, 70), (10 + text_width, 70 + text_height + 10), (0, 0, 0), -1)
            cv2.putText(vis_img, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return vis_img

def process_image(image_path, output_dir=None, visualize=True):
    """Process a single image to detect the T and extract its properties"""
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get T mask
    mask = get_t_mask(img)
    
    # Clean mask
    mask_cleaned = clean_mask(mask)
    
    # Get initial T properties
    initial_props = get_t_properties(mask_cleaned)
    
    # Optimize T parameters if initial detection was successful
    if initial_props is not None:
        properties = optimize_t_parameters_grid_search(mask_cleaned, initial_props)
    else:
        properties = None
    
    # Create output
    result = {
        'success': properties is not None,
        'properties': properties
    }
    
    # Visualize if requested
    if visualize:
        vis_img = visualize_t_detection(img, mask_cleaned, properties)
        
        # Save or display visualization
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(image_path)
            vis_path = os.path.join(output_dir, f"vis_{base_name}")
            plt.figure(figsize=(10, 8))
            plt.imshow(vis_img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(vis_path)
            plt.close()
            
            # Also save the result as JSON
            json_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_result.json")
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            plt.figure(figsize=(10, 8))
            plt.imshow(vis_img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    return result

def process_frame(img, output_dir=None, visualize=True, frame_info=None):
    """Process a frame (as numpy array) to detect the T and extract its properties"""
    # Get T mask
    mask = get_t_mask(img)
    
    # Clean mask
    mask_cleaned = clean_mask(mask)
    
    # Get initial T properties
    initial_props = get_t_properties(mask_cleaned)
    
    # Optimize T parameters if initial detection was successful
    if initial_props is not None:
        properties = optimize_t_parameters_grid_search(mask_cleaned, initial_props)
    else:
        properties = None
    
    # Create output
    result = {
        'success': properties is not None,
        'properties': properties,
        'frame_info': frame_info
    }
    
    # Visualize if requested
    if visualize:
        vis_img = visualize_t_detection(img, mask_cleaned, properties)
        
        # Save or display visualization
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename based on frame info
            if frame_info:
                filename = f"episode_{frame_info['episode_id']:06d}_frame_{frame_info['frame_index']:06d}"
            else:
                filename = f"frame_{np.random.randint(0, 1000000):06d}"
            
            vis_path = os.path.join(output_dir, f"{filename}.png")
            plt.figure(figsize=(10, 8))
            plt.imshow(vis_img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(vis_path)
            plt.close()
            
            # Also save the result as JSON
            json_path = os.path.join(output_dir, f"{filename}_result.json")
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            plt.figure(figsize=(10, 8))
            plt.imshow(vis_img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    return result

def process_dataset_frame(dataset_name="ellen2imagine/pusht_green1", episode_id=0, frame_index=0, 
                          output_dir=None, visualize=True):
    """
    Process a specific frame from a dataset episode.
    
    Args:
        dataset_name (str): Name of the Hugging Face dataset
        episode_id (int): Episode ID to process
        frame_index (int): Frame index within the episode to process
        output_dir (str): Directory to save output files
        visualize (bool): Whether to generate visualization
        
    Returns:
        dict: Processing results
    """
    # Initialize the dataset loader
    loader = DatasetLoader(dataset_name)
    
    # Get episode data
    episode_data = loader.get_episode_data(episode_id)
    
    if not episode_data or 'video_path' not in episode_data or not episode_data['video_path']:
        print(f"Error: Could not load video for episode {episode_id}")
        return {'success': False, 'error': 'Video not found'}
    
    # Extract the requested frame
    frame = extract_frame_from_video(episode_data['video_path'], frame_index)
    
    if frame is None:
        return {'success': False, 'error': 'Frame extraction failed'}
    
    # Process the frame
    frame_info = {
        'dataset': dataset_name,
        'episode_id': episode_id,
        'frame_index': frame_index
    }
    
    return process_frame(frame, output_dir, visualize, frame_info)

def process_frame_sequence(dataset_name="ellen2imagine/pusht_green1", episode_id=0, 
                          start_frame=0, num_frames=10, skip_frames=5,
                          output_dir=None, visualize=True):
    """
    Process a sequence of frames from a dataset episode.
    
    Args:
        dataset_name (str): Name of the Hugging Face dataset
        episode_id (int): Episode ID to process
        start_frame (int): Starting frame index
        num_frames (int): Number of frames to process
        skip_frames (int): Number of frames to skip between each processed frame
        output_dir (str): Directory to save output files
        visualize (bool): Whether to generate visualization
        
    Returns:
        list: List of processing results for each frame
    """
    # Initialize the dataset loader
    loader = DatasetLoader(dataset_name)
    
    # Get frame sequence
    sequence_data = loader.get_frame_sequence(
        episode_id, start_frame, num_frames, skip_frames
    )
    
    if not sequence_data or not sequence_data['frames']:
        print(f"Error: Could not load frames for episode {episode_id}")
        return []
    
    # Process each frame
    results = []
    
    for i, (frame, info) in enumerate(zip(sequence_data['frames'], sequence_data['frame_info'])):
        print(f"Processing frame {i+1}/{len(sequence_data['frames'])}: {info['frame_index']} from episode {episode_id}")
        
        # Process the frame
        result = process_frame(frame, output_dir, visualize, info)
        results.append(result)
    
    # Create a summary file if output directory is specified
    if output_dir:
        summary = {
            'dataset': dataset_name,
            'episode_id': episode_id,
            'start_frame': start_frame,
            'num_frames': num_frames,
            'skip_frames': skip_frames,
            'results': results
        }
        
        summary_path = os.path.join(output_dir, f"sequence_episode_{episode_id:06d}_summary.json")
        with open(summary_path, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            import json
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
            
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    return results

@click.command()
@click.option('--image', '-i', help='Path to input image')
@click.option('--dataset', '-d', default="ellen2imagine/pusht_green1", help='Dataset name')
@click.option('--episode', '-e', default=0, type=int, help='Episode ID')
@click.option('--frame', '-f', default=0, type=int, help='Frame index')
@click.option('--sequence/--no-sequence', default=False, help='Process a sequence of frames')
@click.option('--start', '-s', default=0, type=int, help='Starting frame index for sequence')
@click.option('--num-frames', '-n', default=10, type=int, help='Number of frames to process in sequence')
@click.option('--skip', '-k', default=5, type=int, help='Number of frames to skip between each processed frame')
@click.option('--output', '-o', default=None, help='Output directory for visualization and results')
@click.option('--visualize/--no-visualize', default=True, help='Whether to generate visualization')
def main(image, dataset, episode, frame, sequence, start, num_frames, skip, output, visualize):
    """Detect T-shaped object in an image or video frame and determine its position and orientation"""
    if image:
        print(f"Processing image: {image}")
        result = process_image(image, output, visualize)
        
        if result['success']:
            props = result['properties']
            print(f"T detected at position: ({props['center_x']:.1f}, {props['center_y']:.1f})")
            print(f"T orientation: {props['angle_deg']:.1f} degrees")
        else:
            print("Failed to detect T in the image")
    
    elif sequence:
        print(f"Processing sequence of {num_frames} frames from episode {episode} in dataset {dataset}")
        print(f"Starting at frame {start}, skipping {skip} frames between each processed frame")
        results = process_frame_sequence(
            dataset, episode, start, num_frames, skip, output, visualize
        )
        
        # Print summary of results
        success_count = sum(1 for r in results if r['success'])
        print(f"Processed {len(results)} frames, detected T in {success_count} frames")
        
        if success_count > 0:
            # Calculate average position and orientation
            positions_x = [r['properties']['center_x'] for r in results if r['success']]
            positions_y = [r['properties']['center_y'] for r in results if r['success']]
            angles = [r['properties']['angle_deg'] for r in results if r['success']]
            
            avg_x = sum(positions_x) / len(positions_x)
            avg_y = sum(positions_y) / len(positions_y)
            avg_angle = sum(angles) / len(angles)
            
            print(f"Average T position: ({avg_x:.1f}, {avg_y:.1f})")
            print(f"Average T orientation: {avg_angle:.1f} degrees")
    
    else:
        print(f"Processing frame {frame} from episode {episode} in dataset {dataset}")
        result = process_dataset_frame(dataset, episode, frame, output, visualize)
        
        if result['success']:
            props = result['properties']
            print(f"T detected at position: ({props['center_x']:.1f}, {props['center_y']:.1f})")
            print(f"T orientation: {props['angle_deg']:.1f} degrees")
        else:
            print("Failed to detect T in the image")
    
    print("Done!")

if __name__ == '__main__':
    main()