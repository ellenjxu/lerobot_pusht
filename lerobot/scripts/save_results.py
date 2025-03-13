import os
import json
import cv2
import numpy as np

def save_results(frames, log_data, output_name, total_time, fps, inference_time_s, device):
    log_path = f"{output_name}.json"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    log_summary = {
        "total_time": total_time,
        "fps": fps,
        "intended_duration": inference_time_s,
        "frames_count": len(frames),
        "device": device,
        "frames_data": log_data
    }

    with open(log_path, "w") as f:
        json.dump(log_summary, f, indent=2)

    print(f"Saved log data to {log_path}")

    last_frame = frames[-1]

    last_frame_bgr = cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_name}.png", last_frame_bgr)
    print(f"Saved eval{i}.png")

    video_path = f"{output_name}.mp4"
    height, width = frames[0].shape[1], frames[0].shape[2]

    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()
    print(f"Saved video to {video_path}")
