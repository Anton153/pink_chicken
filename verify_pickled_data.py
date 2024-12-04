import os
import pickle
import cv2
import numpy as np

# Directory containing processed pickle files
test_video_dir = '/home/fizzer/ENPH353_Competition/src/pink_chicken/pipeline_testing/'
output_video_dir = '/home/fizzer/ENPH353_Competition/src/pink_chicken/generated_videos/'

# Create output directory if it doesn't exist
os.makedirs(output_video_dir, exist_ok=True)

# Get a list of all pickle files in the directory
pickle_files = [f for f in os.listdir(test_video_dir) if f.endswith('.pkl')]

for pickle_file in pickle_files:
    try:
        pickle_path = os.path.join(test_video_dir, pickle_file)
        with open(pickle_path, "rb") as f:
            print(f'Opening pickle: {pickle_file}')
            train_data = pickle.load(f)

        # Check structure of train_data
        if not isinstance(train_data, dict) or 'frames' not in train_data or 'commands' not in train_data:
            print(f"Invalid structure in pickle file: {pickle_file}. Skipping.")
            continue

        frames = train_data["frames"]
        commands = train_data["commands"]

        # Check dimensions of frames and commands
        if len(frames) == 0 or len(commands) == 0:
            print(f"No data in pickle file: {pickle_file}. Skipping.")
            continue

        # Video properties
        frame_height, frame_width = frames[0].shape[:2]
        fps = 30  # Frames per second for the output video
        video_filename = os.path.splitext(pickle_file)[0] + '.avi'
        output_video_path = os.path.join(output_video_dir, video_filename)

        # Define the video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        for i, frame in enumerate(frames):
            # Convert frame back to 8-bit format (0-255)
            frame_8bit = (frame * 255).astype(np.uint8)

            # Overlay the command text
            linear_velocity_text = f"Linear Vel: {commands[i, 1]:.2f}"
            angular_velocity_text = f"Angular Vel: {commands[i, 2]:.2f}"
            cv2.putText(
                frame_8bit,
                linear_velocity_text,
                (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA 
            )
            cv2.putText(
                frame_8bit,
                angular_velocity_text,
                (5, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Write the frame to the video file
            video_writer.write(frame_8bit)

        # Release the video writer
        video_writer.release()
        print(f"Video saved to {output_video_path}")

    except Exception as e:
        print(f"Error processing {pickle_file}: {e}")

print("Finished generating videos for all pickle files.")
