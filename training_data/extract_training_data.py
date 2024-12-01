import rosbag
import os
import csv
import cv2 as cv
from cv_bridge import CvBridge
from datetime import datetime  # Import datetime module

# Get the current timestamp
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

# Define the correct output directories
output_dir_vel = '/home/fizzer/ENPH353_Competition/src/pink_chicken/training_data/commands'
output_dir_vid = '/home/fizzer/ENPH353_Competition/src/pink_chicken/training_data/images'

commands_file = os.path.join(output_dir_vel,f"commands_{current_time}.csv")
video_file = os.path.join(output_dir_vid, f"video_output_{current_time}.mp4")  # Change to .mp4

# Ensure the output directories exist
os.makedirs(output_dir_vel, exist_ok=True)
os.makedirs(output_dir_vid, exist_ok=True)

# Initialize CV Bridge
bridge = CvBridge()

# Open the ROS bag file
bag = rosbag.Bag('/home/fizzer/ENPH353_Competition/src/pink_chicken/training_data/driving_data.bag')

# Initialize variables for video writing
fps = 30  # Frames per second
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # MP4 codec
video_writer = None  # Placeholder for dynamic initialization

# Process the /rrbot/camera1/image_raw topic
frame_count = 0
timestamps = []  # Store timestamps for video frames

# make the video
for topic, msg, t in bag.read_messages(topics=['/B1/rrbot/camera1/image_raw']):
    # Convert ROS Image message to OpenCV image
    print(f"Processing topic: {topic}")
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

    # Dynamically initialize VideoWriter with correct frame size
    if video_writer is None:
        frame_height, frame_width, _ = cv_image.shape
        print(f"Frame dimensions: {frame_width}x{frame_height}")
        video_writer = cv.VideoWriter(video_file, fourcc, fps, (frame_width, frame_height))

    # Write the frame to the video file
    print(f"Writing frame {frame_count} to video")
    video_writer.write(cv_image)
    timestamps.append((frame_count, t.to_sec()))
    frame_count += 1
print(f"Total frames processed: {frame_count}")

# Release resources
if video_writer is not None:
    video_writer.release()

print(f"Video saved to: {video_file}")

# Create the CSV file in the specified directory
with open(commands_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['timestamp', 'linear_velocity', 'angular_velocity'])

    # Process only the /B1/cmd_vel topic
    for topic, msg, t in bag.read_messages(topics=['/B1/cmd_vel']):
        linear_velocity = msg.linear.x
        angular_velocity = msg.angular.z
        writer.writerow([t.to_sec(), linear_velocity, angular_velocity])

print(f"Commands saved to: {commands_file}")

# Save video frame timestamps
timestamps_file = os.path.join(output_dir_vid, f"timestamps_{current_time}.csv")
with open(timestamps_file, mode='w', newline='') as ts_file:
    ts_writer = csv.writer(ts_file)
    ts_writer.writerow(['frame_number', 'timestamp'])
    ts_writer.writerows(timestamps)
 
print(f"Timestamps for video frames saved to: {timestamps_file}")

# Close the bag after all processing is complete
bag.close()
