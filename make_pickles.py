import cv2
import numpy as np
import pandas as pd
import os
import pickle

# Define directories
video_dir = '/home/fizzer/ENPH353_Competition/src/pink_chicken/training_data/images'
commands_dir = '/home/fizzer/ENPH353_Competition/src/pink_chicken/training_data/commands'
output_dir = '/home/fizzer/ENPH353_Competition/src/pink_chicken/pipeline_testing'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get a list of all .mp4 video files and corresponding timestamp CSVs
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
timestamp_files = {f: os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.startswith('timestamps_') and f.endswith('.csv')}

# Frame preprocessing parameters
img_size = (128, 128)  # Resize dimensions
max_sync_gap = 0.5  # Maximum time gap for velocity synchronization (in seconds)

# Process each video
for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    print(f"Processing video: {video_file}")

    # Find the corresponding timestamp CSV
    base_name = os.path.splitext(video_file)[0].split('_')[-1]
    timestamp_file = next((path for name, path in timestamp_files.items() if base_name in name), None)

    if not timestamp_file:
        print(f"No timestamp file found for video: {video_file}. Skipping.")
        continue

    try:
        # Load frame timestamps 
        frame_timestamps = pd.read_csv(timestamp_file)
        if 'frame_number' not in frame_timestamps.columns or 'timestamp' not in frame_timestamps.columns:
            print(f"Invalid timestamp file format for {video_file}. Skipping.")
            continue
        frame_timestamps['timestamp'] = pd.to_numeric(frame_timestamps['timestamp'], errors='coerce')

        # Load commands
        timestamp = "_".join(os.path.splitext(video_file)[0].split('_')[-2:])
        commands_file = os.path.join(commands_dir, f"commands_{timestamp}.csv")
        if not os.path.exists(commands_file):
            print(f"No commands file found for video: {video_file}. Skipping.")
            continue

        commands = pd.read_csv(commands_file)
        commands['timestamp'] = pd.to_numeric(commands['timestamp'], errors='coerce')
        commands['linear_velocity'] = pd.to_numeric(commands['linear_velocity'], errors='coerce')
        commands['angular_velocity'] = pd.to_numeric(commands['angular_velocity'], errors='coerce')

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_file}")
            continue

        # Process frames
        frames, frame_data = [], []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_resized = cv2.resize(frame, img_size, interpolation=cv2.INTER_AREA) / 255.0
            frame_timestamp = frame_timestamps.loc[frame_timestamps['frame_number'] == frame_count, 'timestamp']
            if not frame_timestamp.empty:
                frames.append(frame_resized)
                frame_data.append({
                    'timestamp': frame_timestamp.values[0],
                    'frame_index': frame_count
                })
            frame_count += 1

        cap.release()
        print(f"Extracted {len(frames)} frames from video: {video_file}")

        # Align frames with commands
        frame_df = pd.DataFrame(frame_data)
        aligned_data = pd.merge_asof(
            frame_df,
            commands.sort_values('timestamp'),
            on='timestamp',
            direction='backward',
            tolerance=max_sync_gap
        )

        # Forward-fill missing commands to ensure all frames have a valid command
        aligned_data['linear_velocity'].fillna(method='ffill', inplace=True)
        aligned_data['angular_velocity'].fillna(method='ffill', inplace=True)

        # Drop frames only if forward-filling is still not sufficient (e.g., at the start of data)
        aligned_data.dropna(subset=['linear_velocity', 'angular_velocity'], inplace=True)

        # Create synchronized data structure
        synchronized_frames = [frames[int(idx)] for idx in aligned_data['frame_index']]
        synchronized_commands = aligned_data[['timestamp', 'linear_velocity', 'angular_velocity']].values

        # Save processed data for this video
        video_data = {
            "frames": np.array(synchronized_frames),
            "commands": synchronized_commands
        }
        with open(os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}.pkl"), 'wb') as f:
            pickle.dump(video_data, f)

        print(f"Saved processed data for {video_file}")

    except Exception as e:
        print(f"Error processing {video_file}: {e}")

print("Finished processing all videos.")
