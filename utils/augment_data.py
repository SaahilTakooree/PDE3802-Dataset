# This script is used to perform data augmentain on images in the training dataset.
# It iterates over all classes subfolders in 'data/train' and applies augmentation
# function to each image. It generates multiple augmented images per original. The augmented
# images are saved back into the same class folder with a modified filename.
#
# The code does the following augmentation:
#   - Random ratation.
#   - Random brightness change.
#   - Adding random noise.
#   - horizontal_flip.
# 
# Augmenting the dataset helps to imrove the model robustness by providing more diverse
# training examples and reducing overfitting.



# Import dependencies.
from pathlib import Path # Import 'Path' for easy and cross-platform file handling.
import random # Import 'Random' for generating random numbers and shuffling data.
import cv2 # Import 'OpenCV' for image processing.
import numpy as np # Import 'Numpy' for handling of image array and adding noise.
from tqdm import tqdm # Import 'tqdm' for displaying a progress bar during augmentation.

# Define the directory that contains the training images.
TRAIN_DIR = Path("data/train")

# Define the image size that all images will be resized to.
TARGET_SIZE = (224, 224)

# Define the number of augmenation to be generated per original image.
AUGMENT_COUNT = 4

# Function to rotate an image by a random angle (between -15 and 15 degrees).
def rotate_image(image):
  angle = random.randint(-15, 15)
  h, w = image.shape[:2]
  M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
  return cv2.warpAffine(image, M, (w, h))

# Function to randomly adjust the brightness of an image.
def change_brightness(image):
  value = random.randint(-50, 50)
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  
  hsv = hsv.astype(np.int16)
  hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
  
  hsv = hsv.astype(np.uint8)
  return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Function to add random noise to an image.
def add_noise(image):
  noise = np.random.randint(0, 30, image.shape, dtype='uint8')
  return cv2.add(image, noise)

# Function to horizontall flip on an image.
def horizontal_flip(image):
  return cv2.flip(image, 1)


# List of all augmenation functions to randomly choose from.
AUG_FUNCTIONS = [rotate_image, change_brightness, add_noise, horizontal_flip]

# Iterate over each class folder in the training dataset.
for class_dir in TRAIN_DIR.iterdir():
  if class_dir.is_dir():
    # Get all image files in the class folder.
    images = list(class_dir.glob("*.*"))
    
    # Iterate over each image with a prgress bar.
    for img_path in tqdm(images, desc=f"Augmenting {class_dir.name}"):
      
      # Skip any file that is not an image.
      if any(str(img_path).lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
        
        # Read the image using OpenCv
        image = cv2.imread(str(img_path))
        
        # Skip the image if it could not be read.
        if image is None:
          continue
        
        # Resize the iamge to the traget size.
        image = cv2.resize(image, TARGET_SIZE)

        # Extract the base filename and extention.
        base_name = img_path.stem
        ext = img_path.suffix

        # Generate multiple random augmentaed images
        for i in range(AUGMENT_COUNT):
          aug_img = image.copy()
          aug_function = random.choice(AUG_FUNCTIONS)
          aug_img = aug_function(aug_img)

          # Save the augmented image with a new file name.
          new_filename = f"{base_name}_aug{i+1}{ext}"
          cv2.imwrite(str(class_dir / new_filename), aug_img)

# Print a completion message.
print("Augmentation completed.")
