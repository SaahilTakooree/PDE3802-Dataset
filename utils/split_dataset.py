# This scirpt is used to split a dateset which is organised by class folder
# into train, validation, and test sets. It uses sklearn's train_test_split.
# It copies images from 'dateset/' into a new 'data/' directory with the following
# structure:
#
# data/
#   train/
#     class(1)/
#     ....
#     class(n)/
#   val/
#     class(1)/
#     ....
#     class(n)/
#   test/
#     class(1)/
#     ....
#     class(n)/
#
# This was done to organise the dataset into seperate slipts so that the model can
# be trained, validate, and tested properly. The slitting is done randomly to ensure
# a representative distribution of image into each subset. This helps to prevents bias
# allows reliable evaluation of model.



# Import dependencies.
import random # Import 'Random' for generating random numbers and shuffling data.
from pathlib import Path # Import 'PathLib' for easy and cross-platform file handling.
from sklearn.model_selection import train_test_split # Import 'SkLearn' to split datasets into train/val/test
import shutil # Import 'Shutil' for copying files and handling file operations.


# Set a fix random seed for reproducibiliy of dataset splitting.
random.seed(100)

# Define the source dataset folder which contain class subfolders.
source = Path("dataset")

# Define the destination folder where split dataset will be saved.
destination = Path("data")

# Define the proportion of image for training, validation, and testing.
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6 # Ensure that thre ratios sum up to 1.

# Check if the destination folder already exists and warn the user.
if destination.exists():
  print("WARNING: destination exists. Remove/backup if you want fresh split.")
destination.mkdir(parents = True, exist_ok = True) # Create the destination folder, if it does not exist.

# Function to create subdirectories for each class in a givent split.
def make_dirs(split, classes):
  for c in classes:
    (destination / split / c).mkdir(parents = True, exist_ok = True) # Create folder for each class into the split folder.

# Initialise the lists to store folder paths and class names.
folders = []
classes = []

# Get all items in the source dataset folder.
all_items = source.iterdir()

# Filter only directories from the source.
for item in all_items:
  if item.is_dir():
    folders.append(item)

# Extract the class names from the folder names.
for folder in folders:
  classes.append(folder.name)
  
# Create the train, validations, and test folders for each class.
make_dirs("train", classes)
make_dirs("val", classes)
make_dirs("test", classes)

# Iterate over each class to split its images.
for img_class in classes:
  
  # List to store paths of all images in the current class.
  imgs = []
  
  # Get all images files inside the class folder.
  images = list((source/img_class).glob("*"))
  
  # Convert the Path objects to string paths.
  for p in images:
    imgs.append(str(p))

  # Split images into train, val and test sets.
  train_and_val, test = train_test_split(imgs, test_size=test_ratio, random_state=100, stratify=None)
  val_size = val_ratio / (train_ratio + val_ratio)
  train, val = train_test_split(train_and_val, test_size=val_size, random_state=100)

  # Organise the splits into a dictionary for easier iteratin.
  splits = {
    "train": train,
    "val": val,
    "test": test
  }
  
  # Iterate over each split and copy images to the destination.
  for split_name, image_list in splits.items():
    
    # Iterate over each image in the current split.
    for image_path in image_list:
      
      # Get only the filename from the full path.
      image_name = Path(image_path).name
      
      # Determine the destination folder for current image.
      dest_folder = destination / split_name / img_class
      
      # Determine the full destination path for the image.
      dest_path = dest_folder / image_name
      
      # Copy the image to the destination folder.
      shutil.copy(image_path, dest_path)