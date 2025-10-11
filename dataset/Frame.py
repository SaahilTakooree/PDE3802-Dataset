import cv2
import os

video_path = r"20251011_234928.mp4"
output_folder = "frames"
frame_interval = 13

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = os.path.join(output_folder, f"pencil_{saved_count:04d}.png")
        cv2.imwrite(filename, frame)
        saved_count += 1
        print(f"Saved {filename}")

    frame_count += 1

cap.release()
print(f"Done! {saved_count} frames saved to '{output_folder}'.")
