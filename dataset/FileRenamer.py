import os

# Configuration
folder_path = r"C:\Users\mehar\Downloads\stapler"  # Change this to your folder path
name_prefix = "stapler_"  # Change this to your desired prefix
extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")  # Add more if needed

# Get all files in folder
files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]

# Rename files
for i, filename in enumerate(files, start=1):
    ext = os.path.splitext(filename)[1]  # Keep original extension
    new_name = f"{name_prefix}{i}{ext}"
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)
    print(f"Renamed: {filename} → {new_name}")

print("Renaming complete!")
