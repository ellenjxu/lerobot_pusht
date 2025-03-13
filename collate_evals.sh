#!/bin/bash

# Script to collate T-detection results from existing JSON files
# Fixed to properly handle episode grouping

# Set up directories
ACT_OUTPUT_DIR="./output/t_detection_act"
DIFFUSION_OUTPUT_DIR="./output/t_detection_diffusion"
SUMMARY_FILE="./output/t_detection_summary.json"

# Function to process JSON files in a directory and return results array
process_directory() {
    local dir_path=$1
    local policy_name=$2
    
    echo "Processing $policy_name files in $dir_path..." >&2
    
    # Find all JSON files in the directory
    json_files=$(find "$dir_path" -name "*.json" -type f 2>/dev/null)
    
    if [ -z "$json_files" ]; then
        echo "No JSON files found in $dir_path" >&2
        echo "[]"
        return
    fi
    
    # Initialize results array
    local results=()
    
    # Process each JSON file
    for json_file in $json_files; do
        echo "Processing file: $json_file" >&2
        
        # Extract filename and episode info
        file_name=$(basename "$json_file")
        episode_id=$(echo "$file_name" | grep -o -E '[0-9]+' | head -1 || echo "0")
        
        # Read the file and extract values
        if [ -f "$json_file" ]; then
            # Extract success - default to false if not found
            success=$(grep -o '"success": *true' "$json_file" >/dev/null && echo "true" || echo "false")
            
            # Extract IoU value
            iou=$(grep -o '"iou_with_ground_truth": [0-9.]*' "$json_file" | grep -o '[0-9.]*' || echo "0")
            
            # Extract overlap value
            overlap=$(grep -o '"overlap_score": [0-9.]*' "$json_file" | grep -o '[0-9.]*' || echo "0")
            
            echo "File: $file_name, Episode: $episode_id, Success: $success, IoU: $iou" >&2
            
            # Add to results array
            results+=("{\"file\":\"$file_name\",\"episode_id\":$episode_id,\"success\":$success,\"iou\":$iou,\"overlap\":$overlap}")
        fi
    done
    
    # Check if we found any results
    if [ ${#results[@]} -eq 0 ]; then
        echo "No valid result data found in $dir_path" >&2
        echo "[]"
        return
    fi
    
    # Join the results into a JSON array
    local results_json="["
    for ((i=0; i<${#results[@]}; i++)); do
        results_json+="${results[$i]}"
        if [ $i -lt $((${#results[@]}-1)) ]; then
            results_json+=","
        fi
    done
    results_json+="]"
    
    echo "$results_json"
}

# Create episode breakdown directly from results
create_episode_breakdown() {
    local json_array="$1"
    
    # If empty results, return empty array
    if [ "$json_array" = "[]" ]; then
        echo "[]"
        return
    fi
    
    # Create a temporary file to work with
    local temp_file=$(mktemp)
    echo "$json_array" > "$temp_file"
    
    # Get all unique episode IDs
    local episode_ids=$(jq '[.[].episode_id] | unique | sort' "$temp_file")
    
    # Initialize episodes array
    local episodes_array=()
    
    # Process each episode ID
    for id in $(jq -r '.[]' <<< "$episode_ids"); do
        echo "Processing episode $id breakdown..." >&2
        
        # Get all results for this episode
        local episode_results=$(jq "[.[] | select(.episode_id == $id)]" "$temp_file")
        
        # Calculate episode stats
        local count=$(jq 'length' <<< "$episode_results")
        local success_count=$(jq '[.[] | select(.success == true)] | length' <<< "$episode_results")
        local avg_iou=$(jq '[.[].iou | tonumber] | add / length' <<< "$episode_results")
        local max_iou=$(jq '[.[].iou | tonumber] | max' <<< "$episode_results")
        local avg_overlap=$(jq '[.[].overlap | tonumber] | add / length' <<< "$episode_results")
        
        # Create episode JSON
        local episode_json="{\"episode_id\":$id,\"count\":$count,\"success_count\":$success_count,\"avg_iou\":$avg_iou,\"max_iou\":$max_iou,\"avg_overlap\":$avg_overlap}"
        
        # Add to episodes array
        episodes_array+=("$episode_json")
    done
    
    # Clean up
    rm "$temp_file"
    
    # If no episodes, return empty array
    if [ ${#episodes_array[@]} -eq 0 ]; then
        echo "[]"
        return
    fi
    
    # Join episodes into JSON array
    local episodes_json="["
    for ((i=0; i<${#episodes_array[@]}; i++)); do
        episodes_json+="${episodes_array[$i]}"
        if [ $i -lt $((${#episodes_array[@]}-1)) ]; then
            episodes_json+=","
        fi
    done
    episodes_json+="]"
    
    echo "$episodes_json"
}

# Calculate statistics from results
calculate_stats() {
    local json_array="$1"
    
    # If empty results, return default stats
    if [ "$json_array" = "[]" ]; then
        echo "{\"count\":0,\"success_count\":0,\"avg_iou\":0,\"max_iou\":0,\"min_iou\":0,\"avg_overlap\":0}"
        return
    fi
    
    # Create a temporary file to work with
    local temp_file=$(mktemp)
    echo "$json_array" > "$temp_file"
    
    # Calculate stats
    local count=$(jq 'length' "$temp_file")
    local success_count=$(jq '[.[] | select(.success == true)] | length' "$temp_file")
    local avg_iou=$(jq '[.[].iou | tonumber] | add / length' "$temp_file")
    local max_iou=$(jq '[.[].iou | tonumber] | max' "$temp_file")
    local min_iou=$(jq '[.[].iou | tonumber] | min' "$temp_file")
    local avg_overlap=$(jq '[.[].overlap | tonumber] | add / length' "$temp_file")
    
    # Clean up
    rm "$temp_file"
    
    # Return stats JSON
    echo "{\"count\":$count,\"success_count\":$success_count,\"avg_iou\":$avg_iou,\"max_iou\":$max_iou,\"min_iou\":$min_iou,\"avg_overlap\":$avg_overlap}"
}

# Redirect debug output to stderr or a log file
exec 3>&2
exec 2>/tmp/t_detection_debug.log

# Process each directory
act_results=$(process_directory "$ACT_OUTPUT_DIR" "Act")
diffusion_results=$(process_directory "$DIFFUSION_OUTPUT_DIR" "Diffusion")

# Create episode breakdowns
echo "Creating episode breakdowns..." >&2
act_episodes=$(create_episode_breakdown "$act_results")
diffusion_episodes=$(create_episode_breakdown "$diffusion_results")

# Calculate statistics
echo "Calculating statistics..." >&2
act_stats=$(calculate_stats "$act_results")
diffusion_stats=$(calculate_stats "$diffusion_results")

# Calculate comparison metrics
echo "Calculating comparison metrics..." >&2
iou_diff=$(echo "$act_stats" | jq '.avg_iou') 
diffusion_iou=$(echo "$diffusion_stats" | jq '.avg_iou')
iou_difference=$(echo "$iou_diff - $diffusion_iou" | bc -l | sed 's/^\./0./' | sed 's/^-\./-0./')

act_success_rate=$(echo "$act_stats" | jq '.success_count / .count')
diffusion_success_rate=$(echo "$diffusion_stats" | jq '.success_count / .count')
success_difference=$(echo "$act_success_rate - $diffusion_success_rate" | bc -l)

# Create the summary JSON
summary=$(cat <<EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "policies": {
    "act": {
      "stats": $act_stats,
      "episodes": $act_episodes,
      "results": $act_results
    },
    "diffusion": {
      "stats": $diffusion_stats,
      "episodes": $diffusion_episodes,
      "results": $diffusion_results
    },
    "comparison": {
      "iou_difference": $iou_difference,
      "success_rate_difference": $success_difference
    }
  }
}
EOF
)

# Save summary to file
echo "$summary" > "$SUMMARY_FILE"

# Print summary information to stderr (log)
echo "Summary saved to $SUMMARY_FILE" >&2
echo "Act stats: $(echo "$act_stats" | jq -c .)" >&2
echo "Diffusion stats: $(echo "$diffusion_stats" | jq -c .)" >&2
echo "Act episodes: $(echo "$act_episodes" | jq -c .)" >&2
echo "Diffusion episodes: $(echo "$diffusion_episodes" | jq -c .)" >&2

# Create markdown report
report_file="./output/t_detection_report.md"

# Create markdown report safely
{
  echo "# T-Detection Evaluation Report"
  echo "Generated on: $(date)"
  echo ""
  echo "## Summary Statistics"
  echo ""
  echo "| Policy | Count | Success Rate | Avg IoU | Max IoU | Min IoU | Avg Overlap |"
  echo "|--------|-------|-------------|---------|---------|---------|-------------|"
  echo "| Act | $(echo "$act_stats" | jq '.count') | $(echo "scale=2; $(echo "$act_stats" | jq '.success_count') * 100 / $(echo "$act_stats" | jq '.count')" | bc)% | $(echo "$act_stats" | jq '.avg_iou') | $(echo "$act_stats" | jq '.max_iou') | $(echo "$act_stats" | jq '.min_iou') | $(echo "$act_stats" | jq '.avg_overlap') |"
  echo "| Diffusion | $(echo "$diffusion_stats" | jq '.count') | $(echo "scale=2; $(echo "$diffusion_stats" | jq '.success_count') * 100 / $(echo "$diffusion_stats" | jq '.count')" | bc)% | $(echo "$diffusion_stats" | jq '.avg_iou') | $(echo "$diffusion_stats" | jq '.max_iou') | $(echo "$diffusion_stats" | jq '.min_iou') | $(echo "$diffusion_stats" | jq '.avg_overlap') |"
  echo ""
  echo "## Policy Comparison"
  echo ""
  echo "- IoU Difference (Act - Diffusion): $iou_difference"
  echo "- Success Rate Difference: $(echo "$success_difference * 100" | bc -l)%"
  echo ""
  echo "## Episode Breakdown"
  echo ""
} > "$report_file"

# Add each episode to the report
for id in $(echo "$act_episodes" | jq -r '.[].episode_id'); do
  {
    echo "### Episode $id"
    echo ""
    echo "| Policy | Success Rate | Avg IoU | Max IoU | Avg Overlap |"
    echo "|--------|-------------|---------|---------|-------------|"
    
    # Act episode data
    act_ep=$(echo "$act_episodes" | jq -r ".[] | select(.episode_id == $id)")
    if [ -n "$act_ep" ]; then
      act_ep_count=$(echo "$act_ep" | jq '.count')
      act_ep_success=$(echo "$act_ep" | jq '.success_count')
      act_ep_success_rate=$(echo "scale=2; $act_ep_success * 100 / $act_ep_count" | bc)
      act_ep_avg_iou=$(echo "$act_ep" | jq '.avg_iou')
      act_ep_max_iou=$(echo "$act_ep" | jq '.max_iou')
      act_ep_avg_overlap=$(echo "$act_ep" | jq '.avg_overlap')
      echo "| Act | ${act_ep_success_rate}% | $act_ep_avg_iou | $act_ep_max_iou | $act_ep_avg_overlap |"
    else
      echo "| Act | N/A | N/A | N/A | N/A |"
    fi
    
    # Diffusion episode data
    diff_ep=$(echo "$diffusion_episodes" | jq -r ".[] | select(.episode_id == $id)")
    if [ -n "$diff_ep" ]; then
      diff_ep_count=$(echo "$diff_ep" | jq '.count')
      diff_ep_success=$(echo "$diff_ep" | jq '.success_count')
      diff_ep_success_rate=$(echo "scale=2; $diff_ep_success * 100 / $diff_ep_count" | bc)
      diff_ep_avg_iou=$(echo "$diff_ep" | jq '.avg_iou')
      diff_ep_max_iou=$(echo "$diff_ep" | jq '.max_iou')
      diff_ep_avg_overlap=$(echo "$diff_ep" | jq '.avg_overlap')
      echo "| Diffusion | ${diff_ep_success_rate}% | $diff_ep_avg_iou | $diff_ep_max_iou | $diff_ep_avg_overlap |"
    else
      echo "| Diffusion | N/A | N/A | N/A | N/A |"
    fi
    
    echo ""
  } >> "$report_file"
done

# Add top results section
{
  echo "## Top Results"
  echo ""
  echo "### Act Policy (Best IoU)"
  echo ""
} >> "$report_file"

# Add top 3 Act results
echo "$act_results" | jq -r 'sort_by(.iou | tonumber) | reverse | .[0:3] | .[] | "- " + .file + ": IoU = " + (.iou | tostring)' >> "$report_file"

{
  echo ""
  echo "### Diffusion Policy (Best IoU)"
  echo ""
} >> "$report_file"

# Add top 3 Diffusion results
echo "$diffusion_results" | jq -r 'sort_by(.iou | tonumber) | reverse | .[0:3] | .[] | "- " + .file + ": IoU = " + (.iou | tostring)' >> "$report_file"

# Restore stderr
exec 2>&3

# Return simple success message
echo "Collation complete. Results saved to $SUMMARY_FILE and $report_file"