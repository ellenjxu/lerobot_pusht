#!/bin/bash

# Default values
DATASET="ellen2imagine/pusht_green1"
EPISODE=0
START_FRAME=0
NUM_FRAMES=10
OUTPUT_DIR="./output/t_detection_test"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--dataset)
      DATASET="$2"
      shift 2
      ;;
    -e|--episode)
      EPISODE="$2"
      shift 2
      ;;
    -s|--start)
      START_FRAME="$2"
      shift 2
      ;;
    -n|--num-frames)
      NUM_FRAMES="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process multiple frames
for ((i=0; i<NUM_FRAMES; i++)); do
  FRAME=$((START_FRAME + i))
  echo "Processing frame $FRAME from episode $EPISODE"
  
  # Run the detection
  python -m nathan.get_t_info.mask --dataset "$DATASET" --episode "$EPISODE" --frame "$FRAME" \
    --output "$OUTPUT_DIR"
done

echo "T detection test completed. Results saved to $OUTPUT_DIR" 