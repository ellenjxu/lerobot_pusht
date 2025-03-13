#!/bin/bash

# Default values
EPISODE=0
DATASET="ellen2imagine/pusht_green1"
RENDER=true
SAVE_VIDEO=false
ANALYZE=false
MAPPING="xy_only"
SCALE=1.0
SKIP=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --episode)
      EPISODE="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --no-render)
      RENDER=false
      shift
      ;;
    --save-video)
      SAVE_VIDEO=true
      shift
      ;;
    --analyze)
      ANALYZE=true
      shift
      ;;
    --mapping)
      MAPPING="$2"
      shift 2
      ;;
    --scale)
      SCALE="$2"
      shift 2
      ;;
    --skip)
      SKIP="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --episode N       Episode ID to replay (default: 0)"
      echo "  --dataset NAME    Dataset name (default: ellen2imagine/pusht_green1)"
      echo "  --no-render       Disable rendering"
      echo "  --save-video      Save a video of the replay"
      echo "  --analyze         Analyze the actions instead of replaying them"
      echo "  --mapping METHOD  Action mapping method: simple, xy_only, custom (default: xy_only)"
      echo "  --scale VALUE     Scale factor to apply to actions (default: 1.0)"
      echo "  --skip N          Number of frames to skip between each action (default: 1)"
      echo "  --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set up render and save-video flags
RENDER_FLAG=""
if [ "$RENDER" = false ]; then
  RENDER_FLAG="--no-render"
fi

SAVE_VIDEO_FLAG=""
if [ "$SAVE_VIDEO" = true ]; then
  SAVE_VIDEO_FLAG="--save-video"
fi

# Run the appropriate script
if [ "$ANALYZE" = true ]; then
  echo "Analyzing actions for episode $EPISODE from dataset $DATASET..."
  python nathan/map_actions/analyze_actions.py --episode "$EPISODE" --dataset "$DATASET"
else
  echo "Replaying episode $EPISODE from dataset $DATASET..."
  echo "  Mapping method: $MAPPING"
  echo "  Scale factor: $SCALE"
  echo "  Frame skip: $SKIP"
  
  # Make sure MAPPING is not empty
  if [ -z "$MAPPING" ]; then
    MAPPING="xy_only"
  fi
  
  python nathan/map_actions/replay_dataset.py --episode "$EPISODE" --dataset "$DATASET" \
    $RENDER_FLAG $SAVE_VIDEO_FLAG --mapping "$MAPPING" --scale "$SCALE" --skip "$SKIP"
fi