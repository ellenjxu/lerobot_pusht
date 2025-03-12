#!/bin/bash

# Default values
DATASET="ellen2imagine/pusht_green1"
EPISODE=0
FRAME=0
OUTPUT_DIR="./output"

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
    -f|--frame)
      FRAME="$2"
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

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the mask detection
python -m nathan.get_t_info.mask --dataset "$DATASET" --episode "$EPISODE" --frame "$FRAME" --output "$OUTPUT_DIR"

echo "Results saved to $OUTPUT_DIR" 