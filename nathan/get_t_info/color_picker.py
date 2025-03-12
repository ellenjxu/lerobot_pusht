#!/usr/bin/env python3

import os
import sys
import json
import click
import cv2
import numpy as np
from matplotlib import pyplot as plt
from nathan.loader import DatasetLoader
from nathan.get_t_info.mask import extract_frame_from_video

# Global variables to store selected color
selected_color = None
selected_hsv = None
image_display = None

def on_mouse_click(event, x, y, flags, param):
    """Mouse callback function to capture clicks and get pixel color"""
    global selected_color, selected_hsv, image_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get RGB color at the clicked position
        rgb_color = image_display[y, x, :].tolist()
        selected_color = rgb_color
        
        # Convert to HSV
        hsv_pixel = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)
        selected_hsv = hsv_pixel[0, 0, :].tolist()
        
        # Convert RGB to hex
        hex_color = '#{:02x}{:02x}{:02x}'.format(rgb_color[0], rgb_color[1], rgb_color[2])
        
        # Display color information
        print(f"\nSelected color at position ({x}, {y}):")
        print(f"RGB: {rgb_color}")
        print(f"HSV: {selected_hsv}")
        print(f"Hex: {hex_color}")
        
        # Draw a circle at the selected point
        display_img = image_display.copy()
        cv2.circle(display_img, (x, y), 5, (255, 0, 0), 2)
        
        # Add text with color information
        info_text = f"RGB: {rgb_color}, HSV: {selected_hsv}, Hex: {hex_color}"
        cv2.putText(display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the image with the selected point
        cv2.imshow("Color Picker", display_img)

def save_color_config(hex_color, output_dir="./config"):
    """Save the selected color to a configuration file"""
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        "t_color": hex_color,
        "rgb": selected_color,
        "hsv": selected_hsv
    }
    
    config_path = os.path.join(output_dir, "t_color_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Color configuration saved to {config_path}")
    return config_path

def update_mask_py(hex_color):
    """Update the default color in mask.py"""
    mask_py_path = os.path.join(os.path.dirname(__file__), "mask.py")
    
    if not os.path.exists(mask_py_path):
        print(f"Warning: Could not find mask.py at {mask_py_path}")
        return False
    
    try:
        # Read the file
        with open(mask_py_path, 'r') as f:
            content = f.read()
        
        # Replace the default hex color
        updated_content = content.replace(
            "def get_t_mask(img, hex_color='#4CAF50', tolerance=50):",
            f"def get_t_mask(img, hex_color='{hex_color}', tolerance=50):"
        )
        
        # Write the updated content back
        with open(mask_py_path, 'w') as f:
            f.write(updated_content)
        
        print(f"Updated default color in {mask_py_path} to {hex_color}")
        return True
    
    except Exception as e:
        print(f"Error updating mask.py: {e}")
        return False

@click.command()
@click.option('--dataset', '-d', default="ellen2imagine/pusht_green1", help='Dataset name')
@click.option('--episode', '-e', default=0, type=int, help='Episode ID')
@click.option('--frame', '-f', default=0, type=int, help='Frame index')
@click.option('--update-mask/--no-update-mask', default=True, help='Whether to update mask.py with the selected color')
def main(dataset, episode, frame, update_mask):
    """Interactive color picker to select the T color from a dataset frame"""
    global image_display
    
    print(f"Loading frame {frame} from episode {episode} in dataset {dataset}...")
    
    # Initialize the dataset loader
    loader = DatasetLoader(dataset)
    
    # Get episode data
    episode_data = loader.get_episode_data(episode)
    
    if not episode_data or 'video_path' not in episode_data or not episode_data['video_path']:
        print(f"Error: Could not load video for episode {episode}")
        return
    
    # Extract the requested frame
    frame_img = extract_frame_from_video(episode_data['video_path'], frame)
    
    if frame_img is None:
        print("Error: Frame extraction failed")
        return
    
    # Store the image for the callback function
    image_display = frame_img
    
    # Create a window and set the mouse callback
    cv2.namedWindow("Color Picker")
    cv2.setMouseCallback("Color Picker", on_mouse_click)
    
    # Display the image
    cv2.imshow("Color Picker", frame_img)
    
    print("Click on the T to select its color. Press 'S' to save the selected color, or 'Q' to quit without saving.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to exit without saving
        if key == ord('q'):
            break
        
        # Press 's' to save the selected color
        elif key == ord('s'):
            if selected_color is not None:
                # Convert RGB to hex
                hex_color = '#{:02x}{:02x}{:02x}'.format(selected_color[0], selected_color[1], selected_color[2])
                
                # Save the color configuration
                config_path = save_color_config(hex_color)
                
                # Update mask.py if requested
                if update_mask:
                    update_mask_py(hex_color)
                
                print(f"Selected color {hex_color} saved. You can now use this color for T detection.")
            else:
                print("No color selected yet. Click on the image first.")
    
    # Clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 