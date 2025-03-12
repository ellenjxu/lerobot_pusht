#!/bin/bash

# Default values
DATASET="ellen2imagine/pusht_green1"
EPISODE=0
FRAME=0
UPDATE_MASK=true

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
    --no-update)
      UPDATE_MASK=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Run the color picker
if [ "$UPDATE_MASK" = true ]; then
  python -m nathan.get_t_info.color_picker --dataset "$DATASET" --episode "$EPISODE" --frame "$FRAME" --update-mask
else
  python -m nathan.get_t_info.color_picker --dataset "$DATASET" --episode "$EPISODE" --frame "$FRAME" --no-update-mask
fi 