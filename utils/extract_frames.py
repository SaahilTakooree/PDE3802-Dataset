# This script is use to extract frames from videos at specified interval and save them
# as indivial image file in an output_folder.
#
# The code performs the following steps:
#     - Open the video using OpenCV.
#     - Iterates through all the frames in the video.
#     - Saves a frame ever specified interval in the output folder.
#
# Extracting frames for a videos was done to speed up the dataset creation process. It is 
# much easier and faster to record a video of an object from multiple angles than to take 
# individual photoes of the object manually. This approach allows the generation of images 
# efficiently for training of model.



# Import dependencies.
import os # Import 'os' to get system specific function.
import cv2 # Import 'OpenCV' for image processing and video reading.

# Define the path to the input video file.
video_path = r""

# Define the folder where extracted frames will be saved.
output_folder = "frames"

# Define the interval of frames to save a frame.
frame_interval = 13

# Create the output folder, if it does not already exist.
os.makedirs(output_folder, exist_ok = True)

# Open the video file using OpenCV.
capture = cv2.VideoCapture(video_path)

# Check if video can be open, if not, exit out of the script.
if not capture.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialise counters for total frames read and frames saved.
frame_count = 0
saved_count = 0

# Iterate over each frame of the video.
while True:
    
    # Read the frame.
    ret, frame = capture.read()
    
    # Check if video has ended by seeing if no frame has been return.
    if not ret:
        break

    # Save the frame only if matches the interval conditon.
    if frame_count % frame_interval == 0:
        
        # Define the filename with a consistent numeric format.
        filename = os.path.join(output_folder, f"pencil_{saved_count:04d}.png")
        
        # Write the frame as an image file.
        cv2.imwrite(filename, frame)
        
        # Increment the saved frame counter.
        saved_count += 1

    # Increment the total frame counter.
    frame_count += 1

# Release the video capture object.
capture.release()

# Print completion message.
print(f"{saved_count} frames saved to '{output_folder}'.")