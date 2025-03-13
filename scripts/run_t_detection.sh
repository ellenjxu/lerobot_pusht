#!/bin/bash

# Script to run T-detection on final frames in nathan/act and nathan/diffusion
# to evaluate the final IoU scores of each policy
# Handles both image files (PNG) and video files (MP4)

# Set up output directories
ACT_OUTPUT_DIR="./output/t_detection_act"
DIFFUSION_OUTPUT_DIR="./output/t_detection_diffusion"
SUMMARY_FILE="./output/t_detection_summary.json"

# Create output directories if they don't exist
mkdir -p "$ACT_OUTPUT_DIR"
mkdir -p "$DIFFUSION_OUTPUT_DIR"

# Function to process a directory of images and videos
process_directory() {
    local dir_path=$1
    local output_dir=$2
    local policy_name=$3
    
    echo "Processing $policy_name files in $dir_path..."
    
    # Find all image and video files in the directory
    image_files=$(find "$dir_path" -type f -name "*.png")
    video_files=$(find "$dir_path" -type f \( -name "*.mp4" -o -name "*.avi" \))
    
    # Initialize results array for this policy
    declare -a results_array
    
    # Process image files (final frames saved directly)
    for image_file in $image_files; do
        echo "Processing image: $image_file"
        
        # Get image info
        image_name=$(basename "$image_file")
        episode_id=$(echo "$image_name" | grep -o -E '[0-9]+' | head -1)
        
        # Default episode_id if not found
        if [ -z "$episode_id" ]; then
            episode_id=0
        fi
        
        # Output file path
        output_file="$output_dir/${image_name%.png}_result.json"
        
        # Run T-detection on the image
        echo "Running T-detection on image..."
        python -m scripts.get_t_info.mask \
            --image "$image_file" \
            --output "$output_dir" \
            --no-visualize \
            --visualize \
            --verbose
        
        # Extract IoU from the output JSON if it exists
        if [ -f "$output_file" ]; then
            # Use -r to get raw output without quotes
            iou=$(jq -r '.properties.iou_with_ground_truth // 0' "$output_file")
            success=$(jq -r '.success // false' "$output_file")
            
            # Convert 'true'/'false' string to proper JSON boolean
            if [ "$success" = "true" ]; then
                success_bool=true
            else
                success_bool=false
            fi
            
            # Add to results - ensure proper JSON formatting
            results_array+=("{\"file\": \"$image_name\", \"type\": \"image\", \"episode_id\": $episode_id, \"success\": $success_bool, \"iou\": $iou}")
            
            echo "IoU for $image_name: $iou"
        else
            echo "Warning: No output file generated for $image_name"
            results_array+=("{\"file\": \"$image_name\", \"type\": \"image\", \"episode_id\": $episode_id, \"success\": false, \"iou\": 0}")
        fi
    done
    
    # Process video files
    for video_file in $video_files; do
        echo "Processing video: $video_file"
        
        # Get video info
        video_name=$(basename "$video_file")
        episode_id=$(echo "$video_name" | grep -o -E '[0-9]+' | head -1)
        
        # Default episode_id if not found
        if [ -z "$episode_id" ]; then
            episode_id=0
        fi
        
        # Get frame count
        frame_count=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 "$video_file")
        
        # If frame count is not available, try to estimate it
        if [ -z "$frame_count" ] || [ "$frame_count" = "N/A" ]; then
            frame_count=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=noprint_wrappers=1:nokey=1 "$video_file")
        fi
        
        # If still not available, use a different method
        if [ -z "$frame_count" ] || [ "$frame_count" = "N/A" ]; then
            frame_rate=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "$video_file")
            duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$video_file")
            frame_count=$(echo "$frame_rate * $duration" | bc -l | xargs printf "%.0f")
        fi
        
        # Default frame_count if still empty
        if [ -z "$frame_count" ] || [ "$frame_count" = "N/A" ]; then
            echo "Warning: Could not determine frame count for $video_file. Using default."
            frame_count=1
        fi
        
        # Get the last frame index (0-based)
        last_frame=$((frame_count - 1))
        
        # Ensure last_frame is valid (minimum 0)
        if [ "$last_frame" -lt 0 ]; then
            last_frame=0
        fi
        
        # Output file path for video frame result
        output_file="$output_dir/${video_name%.mp4}_final_result.json"
        
        echo "Extracting final frame ($last_frame) from video..."
        python -m scripts.get_t_info.mask \
            --image "$video_file" \
            --frame "$last_frame" \
            --output "$output_dir" \
            --no-visualize \
            --save-video \
            --verbose
        
        # Extract IoU from the output JSON if it exists
        if [ -f "$output_file" ]; then
            iou=$(jq -r '.properties.iou_with_ground_truth // 0' "$output_file")
            success=$(jq -r '.success // false' "$output_file")
            
            # Convert 'true'/'false' string to proper JSON boolean
            if [ "$success" = "true" ]; then
                success_bool=true
            else
                success_bool=false
            fi
            
            # Add to results - ensure proper JSON formatting
            results_array+=("{\"file\": \"$video_name\", \"type\": \"video\", \"episode_id\": $episode_id, \"success\": $success_bool, \"iou\": $iou}")
            
            echo "Final IoU for $video_name: $iou"
        else
            echo "Warning: No output file generated for $video_name"
            results_array+=("{\"file\": \"$video_name\", \"type\": \"video\", \"episode_id\": $episode_id, \"success\": false, \"iou\": 0}")
        fi
    done
    
    # Only proceed if we have results
    if [ ${#results_array[@]} -eq 0 ]; then
        echo "[]"
        return
    fi
    
    # Join the array elements with commas and wrap in square brackets
    results_json="["
    for ((i=0; i<${#results_array[@]}; i++)); do
        results_json+="${results_array[$i]}"
        if [ $i -lt $((${#results_array[@]}-1)) ]; then
            results_json+=","
        fi
    done
    results_json+="]"
    
    echo "$results_json"
}

# Process each directory
echo "Starting T-detection evaluation..."

act_results=$(process_directory "./nathan/act" "$ACT_OUTPUT_DIR" "Act")
diffusion_results=$(process_directory "./nathan/diffusion" "$DIFFUSION_OUTPUT_DIR" "Diffusion")

# Calculate summary statistics
calculate_stats() {
    local results=$1
    
    # If input is empty or not a valid JSON array, return defaults
    if [ -z "$results" ] || [ "$results" = "[]" ]; then
        echo "{\"count\": 0, \"success_count\": 0, \"avg_iou\": 0, \"max_iou\": 0, \"min_iou\": 0}"
        return
    fi
    
    # Try to extract count safely
    local count=$(echo "$results" | jq 'length')
    if [ -z "$count" ] || [ "$count" = "null" ]; then
        count=0
    fi
    
    # If count is 0, return defaults
    if [ "$count" -eq 0 ]; then
        echo "{\"count\": 0, \"success_count\": 0, \"avg_iou\": 0, \"max_iou\": 0, \"min_iou\": 0}"
        return
    fi
    
    # Calculate statistics safely
    local success_count=$(echo "$results" | jq '[.[] | select(.success == true)] | length')
    if [ -z "$success_count" ] || [ "$success_count" = "null" ]; then
        success_count=0
    fi
    
    local avg_iou=$(echo "$results" | jq '[.[] | .iou] | if length > 0 then add / length else 0 end')
    if [ -z "$avg_iou" ] || [ "$avg_iou" = "null" ]; then
        avg_iou=0
    fi
    
    local max_iou=$(echo "$results" | jq '[.[] | .iou] | if length > 0 then max else 0 end')
    if [ -z "$max_iou" ] || [ "$max_iou" = "null" ]; then
        max_iou=0
    fi
    
    local min_iou=$(echo "$results" | jq '[.[] | .iou] | if length > 0 then min else 0 end')
    if [ -z "$min_iou" ] || [ "$min_iou" = "null" ]; then
        min_iou=0
    fi
    
    echo "{\"count\": $count, \"success_count\": $success_count, \"avg_iou\": $avg_iou, \"max_iou\": $max_iou, \"min_iou\": $min_iou}"
}

# Validate JSON before passing to calculate_stats
validate_json() {
    local json_input="$1"
    if echo "$json_input" | jq -e . >/dev/null 2>&1; then
        echo "$json_input"
    else
        echo "[]"
    fi
}

act_results_valid=$(validate_json "$act_results")
diffusion_results_valid=$(validate_json "$diffusion_results")

act_stats=$(calculate_stats "$act_results_valid")
diffusion_stats=$(calculate_stats "$diffusion_results_valid")

# Create summary JSON
summary=$(cat <<EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "policies": {
    "act": {
      "stats": $act_stats,
      "results": $act_results_valid
    },
    "diffusion": {
      "stats": $diffusion_stats,
      "results": $diffusion_results_valid
    }
  }
}
EOF
)

# Validate final summary before saving
if echo "$summary" | jq -e . >/dev/null 2>&1; then
    # Save summary to file
    echo "$summary" | jq '.' > "$SUMMARY_FILE"
    
    # Print summary to console
    echo "Evaluation complete! Summary:"
    echo "Act policy:"
    echo "$act_stats" | jq '.'
    echo "Diffusion policy:"
    echo "$diffusion_stats" | jq '.'
    echo "Full results saved to $SUMMARY_FILE"
else
    echo "Error: Failed to create valid JSON summary. Results may be incomplete or malformed."
    
    # Create a minimal valid JSON as fallback
    echo '{"timestamp": "'"$(date -u +"%Y-%m-%dT%H:%M:%SZ")"'", "error": "Failed to generate complete results", "policies": {"act": {"stats": {"count": 0, "success_count": 0, "avg_iou": 0, "max_iou": 0, "min_iou": 0}, "results": []}, "diffusion": {"stats": {"count": 0, "success_count": 0, "avg_iou": 0, "max_iou": 0, "min_iou": 0}, "results": []}}}' > "$SUMMARY_FILE"
    
    echo "Fallback summary saved to $SUMMARY_FILE"
fi