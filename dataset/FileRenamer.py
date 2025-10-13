import os

folder_path = r"C:\Users\teena\Documents\Middlesex\Year 3\2025-26 PDE3802 Artificial Intelligence (AI) in Robotics\Coursework 1\PDE3802-Dataset\dataset\erasers"
name_prefix = "eraser_"
extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")

files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]

for i, filename in enumerate(files, start=1):
    ext = os.path.splitext(filename)[1]  # Keep original extension
    new_name = f"{name_prefix}{i}{ext}"
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)
    print(f"Renamed: {filename} → {new_name}")

print("Renaming complete!")
