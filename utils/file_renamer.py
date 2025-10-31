# This script is used to rename all image files in a specify folder by 
# providing a consistent naming pattern with a numeric suffix
#
# The script performs the following steps:
#     - List all images in the target folder with supported extenstions.
#     - Iterates over each file and assigns a new name using a perfix and an index.
#     - Keep the original file extension.
#     - Rename the file on the target folder.
#
# This helps to organised datasets, by ensuring filenames and consistent and sorted sequentially.



# Import dependencies.
import os # Import 'os' to get system specific function.

# Define the folder which contains the images to be renamed.
folder_path = r"dataset/usb_sticks"

# Define the name prefix to use for the renaming of the file.
name_prefix = "usb_sticks_"

# Define the allow extention.
extensions = (".jpg", ".jpeg", ".png", ".bmp")

# Lista ll the files in the folder that matches the allowed extensions.
files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]

# Iterates over each file with an index starting from 1.
for i, filename in enumerate(files, start=1):
    
    # Get the original file extention.
    ext = os.path.splitext(filename)[1]
    
    # Create the new filename using the prefix and index.
    new_name = f"{name_prefix}{i}{ext}"
    
    # Get the full path for the original file.
    old_path = os.path.join(folder_path, filename)
    
    # Get the full path for the new file.
    new_path = os.path.join(folder_path, new_name)
    
    # Rename the file.
    os.rename(old_path, new_path)

# Print a completion message.
print("Renaming complete.")