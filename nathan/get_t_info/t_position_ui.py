#!/usr/bin/env python3
import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import cv2
import json
import argparse
from skimage.measure import label

# Add the root directory to the path
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)

# Import functions from mask.py
from nathan.get_t_info.mask import (
    extract_frame_from_video, get_t_mask, clean_mask, 
    get_t_properties, create_t_shape, calculate_overlap_score,
    visualize_t_detection, process_dataset_frame
)
from nathan.loader import DatasetLoader

class TPositionUI:
    def __init__(self, img, initial_mask=None, initial_props=None):
        self.img = img
        self.mask = initial_mask
        
        # Set default T properties if not provided
        if initial_props is None:
            h, w = img.shape[:2]
            self.props = {
                'center_x': w / 2,
                'center_y': h / 2,
                'angle_deg': 0,
                'size': 1.0
            }
        else:
            # Make sure 'size' is included in the properties
            self.props = initial_props.copy()
            if 'size' not in self.props:
                # If size is missing, add a default value
                self.props['size'] = 1.0
        
        # Create the figure and axes
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(left=0.1, bottom=0.25)
        
        # Display the image
        self.img_display = self.ax.imshow(self.img)
        
        # Create sliders for adjusting T properties
        ax_x = plt.axes([0.25, 0.15, 0.65, 0.03])
        ax_y = plt.axes([0.25, 0.10, 0.65, 0.03])
        ax_angle = plt.axes([0.25, 0.05, 0.65, 0.03])
        ax_size = plt.axes([0.25, 0.01, 0.65, 0.03])
        
        h, w = img.shape[:2]
        
        self.slider_x = Slider(ax_x, 'X Position', 0, w, valinit=self.props['center_x'])
        self.slider_y = Slider(ax_y, 'Y Position', 0, h, valinit=self.props['center_y'])
        self.slider_angle = Slider(ax_angle, 'Angle (deg)', 0, 360, valinit=self.props['angle_deg'])
        self.slider_size = Slider(ax_size, 'Size', 0.1, 3.0, valinit=self.props['size'])
        
        # Connect the sliders to the update function
        self.slider_x.on_changed(self.update)
        self.slider_y.on_changed(self.update)
        self.slider_angle.on_changed(self.update)
        self.slider_size.on_changed(self.update)
        
        # Add buttons for saving and resetting
        ax_save = plt.axes([0.8, 0.20, 0.1, 0.04])
        ax_reset = plt.axes([0.65, 0.20, 0.1, 0.04])
        
        self.button_save = Button(ax_save, 'Save')
        self.button_reset = Button(ax_reset, 'Reset')
        
        self.button_save.on_clicked(self.save)
        self.button_reset.on_clicked(self.reset)
        
        # Initialize the T shape visualization
        self.t_shape_artist = None
        self.iou_text = None
        
        # Update the display
        self.update(None)
        
        # Set title
        self.ax.set_title('Adjust T Position, Orientation, and Size')
        
    def update(self, val):
        # Update properties from sliders
        self.props['center_x'] = self.slider_x.val
        self.props['center_y'] = self.slider_y.val
        self.props['angle_deg'] = self.slider_angle.val
        self.props['size'] = self.slider_size.val
        
        # Create a visualization with the current properties
        vis_img = self.create_visualization()
        
        # Update the image display
        self.img_display.set_data(vis_img)
        
        # Calculate and display IoU if a mask is available
        if self.mask is not None:
            iou = self.calculate_iou()
            if self.iou_text is not None:
                self.iou_text.remove()
            self.iou_text = self.ax.text(10, 30, f'IoU: {iou:.4f}', 
                                         color='white', fontsize=12,
                                         bbox=dict(facecolor='black', alpha=0.7))
        
        self.fig.canvas.draw_idle()
    
    def create_visualization(self):
        """Create a visualization of the image with the T overlay"""
        vis_img = self.img.copy()
        
        # If a mask is available, show it as a green overlay
        if self.mask is not None:
            mask_overlay = np.zeros_like(vis_img)
            mask_overlay[self.mask > 0] = [0, 255, 0]  # Green for the mask
            vis_img = cv2.addWeighted(vis_img, 1.0, mask_overlay, 0.3, 0)
        
        # Create T shape using current properties
        t_shape = create_t_shape(
            self.props['center_x'], 
            self.props['center_y'], 
            self.props['angle_deg'], 
            self.props['size']
        )
        
        # Convert to integer points and reshape for cv2.polylines
        t_points = t_shape[:, [1, 0]].astype(np.int32)  # Swap x,y for OpenCV
        t_points = t_points.reshape((-1, 1, 2))
        
        # Draw T shape with thicker outline for better visibility
        cv2.polylines(vis_img, [t_points], True, (255, 0, 255), 3)  # Thicker magenta outline
        
        # Fill T shape with semi-transparent color
        t_mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=np.uint8)
        cv2.fillPoly(t_mask, [t_points], 255)
        t_overlay = np.zeros_like(vis_img)
        t_overlay[t_mask > 0] = [255, 0, 255]  # Magenta fill
        vis_img = cv2.addWeighted(vis_img, 1.0, t_overlay, 0.4, 0)  # Increased opacity
        
        # Draw center point with larger radius
        cv2.circle(vis_img, (int(self.props['center_x']), int(self.props['center_y'])), 
                   7, (0, 0, 255), -1)  # Larger red dot
        
        # Add text showing current properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        
        # Display current values on the image
        text_lines = [
            f"X: {self.props['center_x']:.1f}",
            f"Y: {self.props['center_y']:.1f}",
            f"Angle: {self.props['angle_deg']:.1f}°",
            f"Size: {self.props['size']:.2f}"
        ]
        
        # Add IoU if mask is available
        if self.mask is not None:
            text_lines.append(f"IoU: {self.calculate_iou():.4f}")
        
        # Position text in top-left corner with background
        y_pos = 30
        for line in text_lines:
            text_size = cv2.getTextSize(line, font, 0.7, 2)[0]
            cv2.rectangle(vis_img, (10, y_pos - 20), (10 + text_size[0], y_pos + 5), bg_color, -1)
            cv2.putText(vis_img, line, (10, y_pos), font, 0.7, text_color, 2)
            y_pos += 30
        
        return vis_img
    
    def calculate_iou(self):
        """Calculate IoU between the current T shape and the mask"""
        # Create a mask for the current T shape
        h, w = self.img.shape[:2]
        t_shape = create_t_shape(
            self.props['center_x'], 
            self.props['center_y'], 
            self.props['angle_deg'], 
            self.props['size']
        )
        
        # Convert to integer points and create a mask
        t_points = t_shape[:, [1, 0]].astype(np.int32)  # Swap x,y for OpenCV
        t_points = t_points.reshape((-1, 1, 2))
        
        model_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Check if points are within image bounds
        if np.any(t_points[:, 0, 0] < 0) or np.any(t_points[:, 0, 0] >= w) or \
           np.any(t_points[:, 0, 1] < 0) or np.any(t_points[:, 0, 1] >= h):
            # Some points are outside the image
            return 0.0
        
        cv2.fillPoly(model_mask, [t_points], 1)
        
        # Calculate intersection and union
        intersection = np.logical_and(self.mask, model_mask).sum()
        union = np.logical_or(self.mask, model_mask).sum()
        
        if union > 0:
            return intersection / union
        else:
            return 0.0
    
    def save(self, event):
        """Save the current T properties to a JSON file"""
        # Create a dictionary with the current properties
        data = {
            'center_x': float(self.props['center_x']),
            'center_y': float(self.props['center_y']),
            'angle_deg': float(self.props['angle_deg']),
            'size': float(self.props['size'])
        }
        
        # If a mask is available, add IoU
        if self.mask is not None:
            data['iou'] = float(self.calculate_iou())
        
        # Generate a filename based on timestamp
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, f"t_properties.json")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved T properties to {file_path}")
            print(f"Saved properties: {data}")
            
            # Add a confirmation message to the plot
            if hasattr(self, 'save_text') and self.save_text is not None:
                self.save_text.remove()
            self.save_text = self.ax.text(10, 90, f'Saved to {os.path.basename(file_path)}', 
                                         color='white', fontsize=12,
                                         bbox=dict(facecolor='green', alpha=0.7))
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error saving file: {e}")
            if hasattr(self, 'save_text') and self.save_text is not None:
                self.save_text.remove()
            self.save_text = self.ax.text(10, 90, f'Error saving: {str(e)}', 
                                         color='white', fontsize=12,
                                         bbox=dict(facecolor='red', alpha=0.7))
            self.fig.canvas.draw_idle()
    
    def reset(self, event):
        """Reset the sliders to their initial values"""
        self.slider_x.reset()
        self.slider_y.reset()
        self.slider_angle.reset()
        self.slider_size.reset()
        self.update(None)

def load_image_from_file(image_path):
    """Load an image from a file"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img_rgb

def main():
    parser = argparse.ArgumentParser(description='Interactive UI for defining T position')
    parser.add_argument('--image', '-i', help='Path to input image')
    parser.add_argument('--dataset', '-d', default="ellen2imagine/pusht_green1", help='Dataset name')
    parser.add_argument('--episode', '-e', default=0, type=int, help='Episode ID')
    parser.add_argument('--frame', '-f', default=0, type=int, help='Frame index')
    parser.add_argument('--detect', action='store_true', help='Detect T in the image')
    parser.add_argument('--load-props', '-l', help='Load initial T properties from JSON file')
    
    args = parser.parse_args()
    
    # Load the image
    if args.image:
        img = load_image_from_file(args.image)
        if img is None:
            return
    else:
        # Load from dataset
        result = process_dataset_frame(args.dataset, args.episode, args.frame, 
                                      output_dir=None, visualize=False)
        
        if not result['success']:
            print("Failed to load frame from dataset")
            return
        
        # Get the frame from the dataset loader
        loader = DatasetLoader(args.dataset)
        episode_data = loader.get_episode_data(args.episode)
        img = extract_frame_from_video(episode_data['video_path'], args.frame)
        
        if img is None:
            print("Failed to extract frame from video")
            return
    
    # Detect T in the image if requested
    mask = None
    initial_props = None
    
    if args.detect:
        # Get T mask
        raw_mask = get_t_mask(img)
        mask = clean_mask(raw_mask)
        
        # Get T properties
        initial_props = get_t_properties(mask)
        
        if initial_props is None:
            print("Warning: Failed to detect T in the image. Starting with default properties.")
    
    # Load properties from file if specified
    if args.load_props:
        if os.path.exists(args.load_props):
            with open(args.load_props, 'r') as f:
                loaded_props = json.load(f)
                initial_props = loaded_props
                print(f"Loaded T properties from {args.load_props}")
        else:
            print(f"Warning: Properties file {args.load_props} not found")
    
    # Create and run the UI
    ui = TPositionUI(img, mask, initial_props)
    plt.show()

if __name__ == '__main__':
    main() 