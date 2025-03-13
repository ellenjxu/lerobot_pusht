"""
This script is used to detect and analyze T-shaped objects in images.
It includes functions to create a mask for the T-shaped object, clean the mask,
extract properties of the T-shaped object, and create a T shape model.

python -m nathan.get_t_info.mask   
python -m nathan.get_t_info.mask --sequence --no-visualize
"""

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
import logging
from skimage.measure import regionprops, label
from skimage.morphology import binary_closing, binary_opening, disk
from nathan.loader import DatasetLoader
import scipy.optimize as optimize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

def get_t_mask(img, hex_color='#1f764f', tolerance=55):
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
        [-h_width, -h_height],
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
    """Calculate how well a T shape overlaps with the mask using IOU instead of just coverage"""
    t_shape = create_t_shape(center_x, center_y, angle_deg, size)
    t_points = t_shape[:, [1, 0]].astype(np.int32)
    t_points = t_points.reshape((-1, 1, 2))
    
    h, w = mask.shape
    model_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Check if points are within image bounds
    if np.any(t_points[:, 0, 0] < 0) or np.any(t_points[:, 0, 0] >= w) or \
       np.any(t_points[:, 0, 1] < 0) or np.any(t_points[:, 0, 1] >= h):
        return 0.0
    
    cv2.fillPoly(model_mask, [t_points], 1)
    
    # Calculate IOU instead of just coverage
    intersection = np.logical_and(mask, model_mask).sum()
    union = np.logical_or(mask, model_mask).sum()
    
    if union > 0:
        iou = intersection / union
        return iou
    else:
        return 0.0

def optimize_t_parameters_grid_search(mask, initial_props=None, prior_frame_props=None, verbose=True):
    """
    Find optimal T parameters using grid search to maximize overlap with the mask.
    Prioritizes using prior frame properties if they provide good coverage.
    
    Args:
        mask: Binary mask of the detected T
        initial_props: Optional initial properties as starting point
        prior_frame_props: Optional properties from the prior frame to use as a reference
        verbose: Whether to print detailed progress information
        
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
    
    # Use fixed size of 0.85 instead of estimating from region dimensions
    fixed_size = 0.85
    
    # If we have prior frame properties, check if they provide good coverage (>= 80%)
    if prior_frame_props is not None:
        prior_center_x = prior_frame_props['center_x']
        prior_center_y = prior_frame_props['center_y']
        prior_angle = prior_frame_props['angle_deg']
        
        # Calculate overlap score using prior frame properties
        prior_score = calculate_overlap_score(mask, prior_center_x, prior_center_y, prior_angle, fixed_size)
        if verbose:
            logger.debug(f"Prior frame overlap score: {prior_score:.4f} (threshold: 0.80)")
        
        # If prior frame properties provide good coverage (>= 80%), use them directly
        if prior_score >= 0.80:
            if verbose:
                logger.debug(f"Using prior frame properties directly (overlap: {prior_score:.4f})")
            
            # Create properties dictionary using prior frame values
            optimized_props = {
                'center_x': prior_center_x,
                'center_y': prior_center_y,
                'angle_deg': prior_angle,
                'area': t_region.area,
                'bbox': t_region.bbox,
                'major_axis_length': t_region.major_axis_length,
                'minor_axis_length': t_region.minor_axis_length,
                'max_distance': t_region.major_axis_length * 0.5,
                'size': fixed_size,
                'overlap_score': prior_score
            }
            
            return optimized_props
        
        # Otherwise, use prior frame properties as starting point for optimization
        if verbose:
            logger.debug(f"Prior frame overlap ({prior_score:.4f}) below threshold, optimizing...")
        initial_center_x = prior_center_x
        initial_center_y = prior_center_y
        initial_angle = prior_angle
    
    # Define search ranges
    # For position, search within a radius around the initial center
    min_row, min_col, max_row, max_col = t_region.bbox
    width = max_col - min_col
    height = max_row - min_row
    search_radius = max(width, height) * 0.3  # 10% of the object size
    
    # Position offsets to try (in pixels)
    position_offsets = np.linspace(-search_radius, search_radius, 9)
    
    # Prepare angles to try
    angle_range = 80  # 20 degrees range for fine search
    
    # Also try the full 90-degree rotations to catch cases where orientation is completely off
    additional_angles = [(initial_angle + rot) % 360 for rot in [0, 90, 180, 270]]

    # Find best parameters
    best_score = -1
    best_params = (initial_center_y, initial_center_x, initial_angle, fixed_size)
    
    # First check the initial parameters
    initial_score = calculate_overlap_score(mask, initial_center_x, initial_center_y, initial_angle, fixed_size)
    if verbose:
        logger.debug(f"Initial parameters overlap score: {initial_score:.4f}")
    
    if initial_score >= 0.90:
        if verbose:
            logger.debug(f"Initial parameters provide good coverage (overlap: {initial_score:.4f}), skipping optimization")
        best_score = initial_score
    else:
        # First do a coarse search with the full 90-degree rotations
        if verbose:
            logger.debug("Performing coarse orientation search...")
        for angle in additional_angles:
            score = calculate_overlap_score(mask, initial_center_x, initial_center_y, angle, fixed_size)
            if verbose:
                logger.debug(f"  Angle {angle:.1f}° - overlap: {score:.4f}")
            if score > best_score:
                best_score = score
                best_params = (initial_center_y, initial_center_x, angle, fixed_size)
        
        # Use the best angle from coarse search as the new initial angle
        initial_angle = best_params[2]
        angles = np.linspace(initial_angle - angle_range, initial_angle + angle_range, 30)
        angles = np.unique(angles % 360)  # Normalize to 0-360
        
        if verbose:
            logger.debug(f"Best angle from coarse search: {initial_angle:.1f}° (overlap: {best_score:.4f})")
        
        # If best score from coarse search is already good enough, skip fine search
        if best_score >= 0.80:
            if verbose:
                logger.debug(f"Coarse search provided good coverage (overlap: {best_score:.4f}), skipping fine search")
        else:
            # Now do the full grid search with position and angle (but fixed size)
            if verbose:
                logger.debug("Performing fine position and orientation search...")
            for y_offset in position_offsets:
                for x_offset in position_offsets:
                    for angle in angles:
                        # Test these parameters
                        center_y = initial_center_y + y_offset
                        center_x = initial_center_x + x_offset
                        
                        score = calculate_overlap_score(mask, center_x, center_y, angle, fixed_size)
                        
                        if score > best_score:
                            best_score = score
                            best_params = (center_y, center_x, angle, fixed_size)
            
            if verbose:
                logger.debug(f"Best parameters after fine search - overlap: {best_score:.4f}")
    
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

def load_ground_truth_properties(json_path=None):
    """
    Load T properties from a JSON file to use as ground truth.
    
    Args:
        json_path (str): Path to the JSON file with T properties
        
    Returns:
        dict: T properties or None if file not found
    """
    if json_path is None:
        # Use default path in the same directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(base_dir, "t_properties.json")
    
    if not os.path.exists(json_path):
        print(f"Warning: Ground truth properties file not found at {json_path}")
        return None
    
    try:
        with open(json_path, 'r') as f:
            properties = json.load(f)
        return properties
    except Exception as e:
        print(f"Error loading ground truth properties: {e}")
        return None

def calculate_iou_with_ground_truth(detected_props, ground_truth_props, img_shape):
    """
    Calculate IoU between detected T and ground truth T.
    
    Args:
        detected_props (dict): Properties of the detected T
        ground_truth_props (dict): Properties of the ground truth T
        img_shape (tuple): Shape of the image (height, width)
        
    Returns:
        float: IoU score
    """
    h, w = img_shape[:2]
    
    # Create mask for detected T
    detected_shape = create_t_shape(
        detected_props['center_x'],
        detected_props['center_y'],
        detected_props['angle_deg'],
        detected_props.get('size', 1.0)
    )
    
    # Convert to integer points and create a mask
    detected_points = detected_shape[:, [1, 0]].astype(np.int32)  # Swap x,y for OpenCV
    detected_points = detected_points.reshape((-1, 1, 2))
    
    detected_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Check if points are within image bounds
    if np.any(detected_points[:, 0, 0] < 0) or np.any(detected_points[:, 0, 0] >= w) or \
       np.any(detected_points[:, 0, 1] < 0) or np.any(detected_points[:, 0, 1] >= h):
        # Some points are outside the image
        return 0.0
    
    cv2.fillPoly(detected_mask, [detected_points], 1)
    
    # Create mask for ground truth T
    gt_shape = create_t_shape(
        ground_truth_props['center_x'],
        ground_truth_props['center_y'],
        ground_truth_props['angle_deg'],
        ground_truth_props.get('size', 1.0)
    )
    
    # Convert to integer points and create a mask
    gt_points = gt_shape[:, [1, 0]].astype(np.int32)  # Swap x,y for OpenCV
    gt_points = gt_points.reshape((-1, 1, 2))
    
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Check if points are within image bounds
    if np.any(gt_points[:, 0, 0] < 0) or np.any(gt_points[:, 0, 0] >= w) or \
       np.any(gt_points[:, 0, 1] < 0) or np.any(gt_points[:, 0, 1] >= h):
        # Some points are outside the image
        return 0.0
    
    cv2.fillPoly(gt_mask, [gt_points], 1)
    
    # Calculate intersection and union
    intersection = np.logical_and(detected_mask, gt_mask).sum()
    union = np.logical_or(detected_mask, gt_mask).sum()

    if union > 0:
        return intersection / union
    else:
        return 0.0

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
        
        # Load ground truth properties and visualize if available
        ground_truth_props = load_ground_truth_properties()
        if ground_truth_props is not None:
            # Create and draw ground truth T shape
            gt_shape = create_t_shape(
                ground_truth_props['center_x'],
                ground_truth_props['center_y'],
                ground_truth_props['angle_deg'],
                ground_truth_props.get('size', 1.0)
            )
            
            # Convert to integer points and reshape for cv2.polylines
            gt_points = gt_shape[:, [1, 0]].astype(np.int32)  # Swap x,y for OpenCV
            gt_points = gt_points.reshape((-1, 1, 2))
            
            # Draw ground truth T shape with a different color
            cv2.polylines(vis_img, [gt_points], True, (0, 255, 255), 2)  # Yellow outline
            
            # Fill ground truth T shape with semi-transparent color
            gt_t_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.fillPoly(gt_t_mask, [gt_points], 255)
            gt_overlay = np.zeros_like(vis_img)
            gt_overlay[gt_t_mask > 0] = [0, 255, 255]  # Yellow fill
            vis_img = cv2.addWeighted(vis_img, 1.0, gt_overlay, 0.2, 0)
            
            # Calculate and display IoU with ground truth
            iou_with_gt = calculate_iou_with_ground_truth(properties, ground_truth_props, img.shape)
            properties['iou_with_ground_truth'] = iou_with_gt
        
        # Draw orientation line along the stem of the T
        angle_deg = properties['angle_deg']
        
        # Calculate the stem direction based on how create_t_shape works
        # When angle is 0, stem points down (270 degrees in standard coordinates)
        stem_angle_rad = np.radians((-angle_deg + 360) % 360)
        
        length = 100  # Length of the orientation line
        
        # Calculate end point of the orientation line
        end_x = int(center_x + length * np.cos(stem_angle_rad))
        end_y = int(center_y + length * np.sin(stem_angle_rad))
        
        # Draw the orientation line
        cv2.line(vis_img, (center_x, center_y), (end_x, end_y), (255, 0, 0), 3)  # Blue line
        
        # Calculate the position of the blue dot at the top of the T
        # The top of the T is in the opposite direction of the stem
        top_angle_rad = np.radians(((-angle_deg + 180) % 360))
        
        # Distance from center to top of T (based on T dimensions in create_t_shape)
        # In create_t_shape, the vertical stem height is 60*size and horizontal bar height is 15*size
        top_distance = 15 * t_size  # This is the half-height of the horizontal bar
        
        # Calculate the position of the top of the T
        top_x = int(center_x + top_distance * np.cos(top_angle_rad))
        top_y = int(center_y + top_distance * np.sin(top_angle_rad))
        
        # Draw the blue dot at the top of the T
        cv2.circle(vis_img, (top_x, top_y), 5, (255, 0, 0), -1)  # Blue dot
        
        # Draw center point (original red dot)
        cv2.circle(vis_img, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot
        
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
        
        # Add top point coordinates
        text = f"Top point: ({top_x:.1f}, {top_y:.1f})"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(vis_img, (10, 70), (10 + text_width, 70 + text_height + 10), (0, 0, 0), -1)
        cv2.putText(vis_img, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Add size information
        if 'size' in properties:
            text = f"Size: {properties['size']:.3f}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(vis_img, (10, 100), (10 + text_width, 100 + text_height + 10), (0, 0, 0), -1)
            cv2.putText(vis_img, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Add IoU with ground truth if available
        if 'iou_with_ground_truth' in properties:
            text = f"IoU with GT: {properties['iou_with_ground_truth']:.3f}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(vis_img, (10, 130), (10 + text_width, 130 + text_height + 10), (0, 0, 0), -1)
            cv2.putText(vis_img, text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Add a legend for the colors
            legend_y = 180
            # Detected T (magenta)
            cv2.rectangle(vis_img, (10, legend_y), (30, legend_y + 20), (255, 0, 255), -1)
            cv2.putText(vis_img, "Detected T", (40, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # Ground truth T (yellow)
            cv2.rectangle(vis_img, (10, legend_y + 30), (30, legend_y + 50), (0, 255, 255), -1)
            cv2.putText(vis_img, "Ground Truth T", (40, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # Red dot (center)
            cv2.circle(vis_img, (20, legend_y + 70), 5, (0, 0, 255), -1)
            cv2.putText(vis_img, "Center Point", (40, legend_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # Blue dot (top of T)
            cv2.circle(vis_img, (20, legend_y + 100), 5, (255, 0, 0), -1)
            cv2.putText(vis_img, "Top of T", (40, legend_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis_img

def process_image(image_path, output_dir=None, visualize=True, verbose=True):
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
        properties = optimize_t_parameters_grid_search(mask_cleaned, initial_props, verbose=verbose)
        
        # Load ground truth properties and calculate IoU
        ground_truth_props = load_ground_truth_properties()
        if ground_truth_props is not None:
            iou_with_gt = calculate_iou_with_ground_truth(properties, ground_truth_props, img.shape)
            properties['iou_with_ground_truth'] = iou_with_gt
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

def process_frame(img, output_dir=None, visualize=True, frame_info=None, prior_frame_props=None, verbose=True):
    """Process a frame (as numpy array) to detect the T and extract its properties"""
    # Get T mask
    mask = get_t_mask(img)
    
    # Clean mask
    mask_cleaned = clean_mask(mask)
    
    # Get initial T properties
    initial_props = get_t_properties(mask_cleaned)
    
    # Optimize T parameters if initial detection was successful
    if initial_props is not None:
        properties = optimize_t_parameters_grid_search(mask_cleaned, initial_props, prior_frame_props, verbose=verbose)
        
        # Load ground truth properties and calculate IoU
        ground_truth_props = load_ground_truth_properties()
        if ground_truth_props is not None:
            iou_with_gt = calculate_iou_with_ground_truth(properties, ground_truth_props, img.shape)
            properties['iou_with_ground_truth'] = iou_with_gt
    else:
        properties = None
    
    # Create output
    result = {
        'success': properties is not None,
        'properties': properties,
        'frame_info': frame_info
    }
    
    # Always save detailed data if output directory is specified
    if output_dir is not None and properties is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename based on frame info
        if frame_info:
            filename = f"episode_{frame_info['episode_id']:06d}_frame_{frame_info['frame_index']:06d}"
        else:
            filename = f"frame_{np.random.randint(0, 1000000):06d}"
        
        # Save JSON data
        json_path = os.path.join(output_dir, f"{filename}_result.json")
        
        # Ensure all numpy values are converted to Python native types
        frame_data = {
            'success': result['success'],
            'frame_info': frame_info,
            'properties': {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else 
                   int(v) if isinstance(v, (np.int32, np.int64)) else
                   v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in properties.items()
            } if properties else None
        }
        
        with open(json_path, 'w') as f:
            json.dump(frame_data, f, indent=2)
    
    # Visualize if requested
    if visualize:
        vis_img = visualize_t_detection(img, mask_cleaned, properties)
        
        # Save or display visualization
        if output_dir is not None:
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
                          start_frame=0, num_frames=None, skip_frames=5,
                          output_dir=None, visualize=True, visualize_pause=False, save_video=True, verbose=False):
    """
    Process a sequence of frames from a dataset episode.
    
    Args:
        dataset_name (str): Name of the Hugging Face dataset
        episode_id (int): Episode ID to process
        start_frame (int): Starting frame index
        num_frames (int, optional): Number of frames to process. If None, process all frames.
        skip_frames (int): Number of frames to skip between each processed frame
        output_dir (str): Directory to save output files
        visualize (bool): Whether to display visualizations
        visualize_pause (bool): Whether to display visualizations and wait for user to close each window
        save_video (bool): Whether to save a video of the sequence
        verbose (bool): Whether to print detailed debug information
        
    Returns:
        list: List of processing results for each frame
    """
    # Initialize the dataset loader
    loader = DatasetLoader(dataset_name)
    
    # Log save_video flag to help debug
    logger.info(f"Save video flag is set to: {save_video}")
    
    # Get episode data to determine total frames if num_frames is None
    if num_frames is None:
        episode_data = loader.get_episode_data(episode_id)
        if not episode_data or 'video_path' not in episode_data or not episode_data['video_path']:
            logger.error(f"Could not load video for episode {episode_id}")
            return []
            
        # Open the video to get frame count
        cap = cv2.VideoCapture(episode_data['video_path'])
        if not cap.isOpened():
            logger.error(f"Could not open video {episode_data['video_path']}")
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Calculate how many frames to process
        num_frames = (total_frames - start_frame + skip_frames - 1) // skip_frames
        logger.info(f"Video has {total_frames} frames. Processing {num_frames} frames starting from frame {start_frame} with skip {skip_frames}.")
    
    # Get frame sequence
    sequence_data = loader.get_frame_sequence(
        episode_id, start_frame, num_frames, skip_frames
    )
    
    if not sequence_data or not sequence_data['frames']:
        logger.error(f"Could not load frames for episode {episode_id}")
        return []
    
    # Process each frame
    results = []
    prior_frame_props = None
    visualization_frames = []
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Process all frames first
    for i, (frame, info) in enumerate(zip(sequence_data['frames'], sequence_data['frame_info'])):
        logger.info(f"Processing frame {i+1}/{len(sequence_data['frames'])}: {info['frame_index']} from episode {episode_id}")
        
        # Process the frame, passing the prior frame's properties
        result = process_frame(frame, None, False, info, prior_frame_props, verbose=verbose)
        results.append(result)
        
        # Update prior_frame_props for the next iteration if this frame was successful
        if result['success']:
            prior_frame_props = result['properties']
        
        # Create visualization for this frame (for video or display)
        mask = get_t_mask(frame)
        mask_cleaned = clean_mask(mask)
        vis_img = visualize_t_detection(frame, mask_cleaned, result['properties'] if result['success'] else None)
        
        # Store visualization frame for video
        if save_video:
            visualization_frames.append(vis_img)
        
        # Display visualization if requested
        if visualize:
            plt.figure(figsize=(10, 8))
            plt.imshow(vis_img)
            plt.title(f"Frame {info['frame_index']} - {'Close window to continue' if visualize_pause else ''}")
            plt.axis('off')
            plt.tight_layout()
            
            if visualize_pause:
                plt.show(block=True)  # Block until the user closes the window
            else:
                plt.show(block=False)  # Don't block, just display and continue
                plt.pause(0.1)  # Small pause to ensure the window appears
                plt.close()
    
    # Now save the video with all visualization frames
    if save_video and output_dir and visualization_frames:
        logger.info(f"Preparing to save video with {len(visualization_frames)} frames")
        
        # Use a more compatible codec and container format
        video_path = os.path.join(output_dir, f"episode_{episode_id:06d}_sequence.mp4")
        
        try:
            # Get frame dimensions
            h, w = visualization_frames[0].shape[:2]
            
            # Try different codec options for better compatibility
            # Option 1: H.264 codec with MP4 container
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            video_writer = cv2.VideoWriter(video_path, fourcc, 10, (w, h))
            
            if not video_writer.isOpened():
                logger.warning(f"Failed to open video writer with avc1 codec, trying mp4v...")
                # Option 2: MP4V codec
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, 10, (w, h))
                
                if not video_writer.isOpened():
                    logger.warning(f"Failed to open video writer with mp4v codec, trying XVID...")
                    # Option 3: XVID codec with AVI container
                    video_path = os.path.join(output_dir, f"episode_{episode_id:06d}_sequence.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 10, (w, h))
            
            if not video_writer.isOpened():
                logger.error(f"Failed to open any video writer")
            else:
                # Write all frames to video
                for frame_idx, vis_img in enumerate(visualization_frames):
                    # Convert from RGB to BGR for OpenCV
                    vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                    video_writer.write(vis_img_bgr)
                
                # Release video writer
                video_writer.release()
                
                # Check if the video file was created successfully
                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    logger.info(f"Successfully saved visualization video to: {os.path.abspath(video_path)}")
                    
                    # For MP4 files, try to run ffmpeg to ensure compatibility
                    if video_path.endswith('.mp4'):
                        try:
                            import subprocess
                            temp_path = video_path + ".temp.mp4"
                            cmd = [
                                'ffmpeg', '-y', '-i', video_path, 
                                '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                                '-strict', 'experimental', temp_path
                            ]
                            logger.info(f"Running ffmpeg to ensure video compatibility: {' '.join(cmd)}")
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            
                            if result.returncode == 0 and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                                os.replace(temp_path, video_path)
                                logger.info(f"Successfully converted video to a more compatible format")
                            else:
                                logger.warning(f"ffmpeg conversion failed: {result.stderr}")
                        except Exception as e:
                            logger.warning(f"Error running ffmpeg: {str(e)}")
                else:
                    logger.error(f"Video file was not created or is empty: {os.path.abspath(video_path)}")
        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")
    elif save_video and output_dir:
        logger.warning("No visualization frames to save to video")
    elif save_video and not output_dir:
        logger.warning("Cannot save video without an output directory")
    else:
        logger.info("Video saving is disabled")
    
    # Create a summary file if output directory is specified
    if output_dir:
        # Create a deep copy of results to avoid modifying the original
        import copy
        summary_results = copy.deepcopy(results)
        
        # Convert numpy values to Python native types for JSON serialization
        for result in summary_results:
            if result['success'] and result['properties']:
                result['properties'] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else 
                       int(v) if isinstance(v, (np.int32, np.int64)) else
                       v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in result['properties'].items()
                }
        
        # Create summary data
        summary = {
            'dataset': dataset_name,
            'episode_id': episode_id,
            'start_frame': start_frame,
            'num_frames': num_frames,
            'skip_frames': skip_frames,
            'metrics': {
                'total_frames': len(results),
                'successful_detections': sum(1 for r in results if r['success'])
            },
            'video_path': os.path.abspath(video_path) if save_video and 'video_path' in locals() else None
        }
        
        # Add average metrics if we have successful detections
        success_count = sum(1 for r in results if r['success'])
        if success_count > 0:
            positions_x = [r['properties']['center_x'] for r in results if r['success']]
            positions_y = [r['properties']['center_y'] for r in results if r['success']]
            angles = [r['properties']['angle_deg'] for r in results if r['success']]
            
            summary['metrics']['average_position_x'] = sum(positions_x) / len(positions_x)
            summary['metrics']['average_position_y'] = sum(positions_y) / len(positions_y)
            summary['metrics']['average_angle_deg'] = sum(angles) / len(angles)
            
            # Add IoU statistics if available
            ious = [r['properties'].get('iou_with_ground_truth', 0) for r in results if r['success'] and 'iou_with_ground_truth' in r['properties']]
            if ious:
                summary['metrics']['average_iou'] = sum(ious) / len(ious)
                summary['metrics']['min_iou'] = min(ious)
                summary['metrics']['max_iou'] = max(ious)
        
        # Add frame-by-frame results
        summary['frame_results'] = [
            {
                'frame_index': r['frame_info']['frame_index'],
                'success': r['success'],
                'center_x': r['properties']['center_x'] if r['success'] else None,
                'center_y': r['properties']['center_y'] if r['success'] else None,
                'angle_deg': r['properties']['angle_deg'] if r['success'] else None,
                'overlap_score': r['properties'].get('overlap_score') if r['success'] else None,
                'iou_with_ground_truth': r['properties'].get('iou_with_ground_truth') if r['success'] else None
            }
            for r in results
        ]
        
        # Save summary JSON
        summary_path = os.path.join(output_dir, f"sequence_episode_{episode_id:06d}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary data to: {os.path.abspath(summary_path)}")
    
    return results

def get_t_position_and_orientation(img, debug=False):
    """
    Extract the position of the top of the T and its orientation from an image.
    
    Args:
        img: Input RGB image
        debug: If True, also return the visualization image
        
    Returns:
        tuple: (x_T, y_T, theta_T) where:
            - (x_T, y_T) is the position of the top of the T
            - theta_T is the orientation angle in degrees
        If debug=True, returns (x_T, y_T, theta_T, vis_img)
    """
    # Get T mask
    mask = get_t_mask(img)
    
    # Clean mask
    mask_cleaned = clean_mask(mask)
    
    # Get initial T properties
    initial_props = get_t_properties(mask_cleaned)
    
    # Default return values if detection fails
    x_T, y_T, theta_T = None, None, None
    
    # Optimize T parameters if initial detection was successful
    if initial_props is not None:
        properties = optimize_t_parameters_grid_search(mask_cleaned, initial_props, verbose=False)
        
        if properties is not None:
            # Extract center coordinates and angle
            center_x = properties['center_x']
            center_y = properties['center_y']
            angle_deg = properties['angle_deg']
            t_size = properties.get('size', 1.0)
            
            # Calculate the position of the top of the T
            # The top of the T is in the opposite direction of the stem
            top_angle_rad = np.radians(((-angle_deg + 180) % 360))
            
            # Distance from center to top of T (based on T dimensions in create_t_shape)
            top_distance = 15 * t_size  # This is the half-height of the horizontal bar
            
            # Calculate the position of the top of the T
            x_T = center_x + top_distance * np.cos(top_angle_rad)
            y_T = center_y + top_distance * np.sin(top_angle_rad)
            
            # The orientation angle is the angle of the stem
            theta_T = angle_deg
    
    # If debug is True, also create and return the visualization image
    if debug:
        vis_img = None
        if initial_props is not None and properties is not None:
            vis_img = visualize_t_detection(img, mask_cleaned, properties)
        return x_T, y_T, theta_T, vis_img
    
    return x_T, y_T, theta_T

@click.command()
@click.option('--image', '-i', help='Path to input image')
@click.option('--dataset', '-d', default="ellen2imagine/pusht_green1", help='Dataset name')
@click.option('--episode', '-e', default=0, type=int, help='Episode ID')
@click.option('--frame', '-f', default=0, type=int, help='Frame index')
@click.option('--sequence/--no-sequence', default=True, help='Process a sequence of frames')
@click.option('--start', '-s', default=0, type=int, help='Starting frame index for sequence')
@click.option('--num-frames', '-n', default=None, type=int, help='Number of frames to process in sequence. If not specified, process all frames.')
@click.option('--skip', '-k', default=5, type=int, help='Number of frames to skip between each processed frame')
@click.option('--output', '-o', default='./output/t_detection_sequence', help='Output directory for visualization and results')
@click.option('--visualize/--no-visualize', default=False, help='Whether to display visualizations')
@click.option('--visualize-pause/--no-visualize-pause', default=False, help='Whether to display visualizations and wait for user to close each window')
@click.option('--save-video/--no-save-video', default=True, help='Whether to save a video of the sequence')
@click.option('--ground-truth', '-g', default=None, help='Path to ground truth T properties JSON file')
@click.option('--verbose/--quiet', default=False, help='Control debug output verbosity')
def main(image, dataset, episode, frame, sequence, start, num_frames, skip, output, visualize, visualize_pause, save_video, ground_truth, verbose):
    """Detect T-shaped object in an image or video frame and determine its position and orientation"""
    # Set logging level based on verbosity
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # If visualize_pause is enabled, also enable visualize
    if visualize_pause:
        visualize = True
    
    # Load ground truth properties if specified
    if ground_truth:
        gt_props = load_ground_truth_properties(ground_truth)
        if gt_props:
            logger.info(f"Loaded ground truth T properties from {ground_truth}")
            if verbose:
                logger.info(f"Ground truth position: ({gt_props['center_x']:.1f}, {gt_props['center_y']:.1f})")
                logger.info(f"Ground truth orientation: {gt_props['angle_deg']:.1f} degrees")
                logger.info(f"Ground truth size: {gt_props.get('size', 1.0):.2f}")
    
    if image:
        logger.info(f"Processing image: {image}")
        result = process_image(image, output, visualize, verbose=verbose)
        
        if result['success']:
            props = result['properties']
            logger.info(f"T detected at position: ({props['center_x']:.1f}, {props['center_y']:.1f})")
            logger.info(f"T orientation: {props['angle_deg']:.1f} degrees")
            
            if 'iou_with_ground_truth' in props:
                logger.info(f"IoU with ground truth: {props['iou_with_ground_truth']:.4f}")
        else:
            logger.warning("Failed to detect T in the image")
    
    elif sequence:
        logger.info(f"Processing sequence from episode {episode} in dataset {dataset}")
        logger.info(f"Starting at frame {start}, skipping {skip} frames between each processed frame")
        if num_frames:
            logger.info(f"Processing {num_frames} frames")
        else:
            logger.info("Processing all frames in the video")
        
        results = process_frame_sequence(
            dataset, episode, start, num_frames, skip, output, visualize, visualize_pause, save_video, verbose
        )
        
        # Print summary of results
        success_count = sum(1 for r in results if r['success'])
        logger.info(f"Processed {len(results)} frames, detected T in {success_count} frames")
        
        if success_count > 0:
            # Calculate average position, orientation, and IoU with ground truth
            positions_x = [r['properties']['center_x'] for r in results if r['success']]
            positions_y = [r['properties']['center_y'] for r in results if r['success']]
            angles = [r['properties']['angle_deg'] for r in results if r['success']]
            
            avg_x = sum(positions_x) / len(positions_x)
            avg_y = sum(positions_y) / len(positions_y)
            avg_angle = sum(angles) / len(angles)
            
            logger.info(f"Average T position: ({avg_x:.1f}, {avg_y:.1f})")
            logger.info(f"Average T orientation: {avg_angle:.1f} degrees")
            
            # Print IoU statistics if available
            ious = [r['properties'].get('iou_with_ground_truth', 0) for r in results if r['success'] and 'iou_with_ground_truth' in r['properties']]
            if ious:
                avg_iou = sum(ious) / len(ious)
                min_iou = min(ious)
                max_iou = max(ious)
                logger.info(f"IoU with ground truth - Avg: {avg_iou:.4f}, Min: {min_iou:.4f}, Max: {max_iou:.4f}")
    
    else:
        logger.info(f"Processing frame {frame} from episode {episode} in dataset {dataset}")
        result = process_dataset_frame(dataset, episode, frame, output, visualize)
        
        if result['success']:
            props = result['properties']
            logger.info(f"T detected at position: ({props['center_x']:.1f}, {props['center_y']:.1f})")
            logger.info(f"T orientation: {props['angle_deg']:.1f} degrees")
            
            if 'iou_with_ground_truth' in props:
                logger.info(f"IoU with ground truth: {props['iou_with_ground_truth']:.4f}")
        else:
            logger.warning("Failed to detect T in the image")
    
    logger.info("Done!")

if __name__ == '__main__':
    main()