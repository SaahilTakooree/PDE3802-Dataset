import cv2
import os
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm  # for progress bar

# ==============================
# CONFIGURATION
# ==============================
TRAIN_DIR = Path("data/train")
TARGET_SIZE = (224, 224)
AUGMENT_COUNT = 4  # number of extra images to generate per original

# ==============================
# AUGMENTATION FUNCTIONS
# ==============================

def rotate_image(image):
    angle = random.randint(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def change_brightness(image):
    value = random.randint(-50, 50)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    hsv = hsv.astype(np.int16)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def add_noise(image):
    noise = np.random.randint(0, 30, image.shape, dtype='uint8')
    return cv2.add(image, noise)

def horizontal_flip(image):
    return cv2.flip(image, 1)

# ------------------------------
AUG_FUNCTIONS = [rotate_image, change_brightness, add_noise, horizontal_flip]
# ------------------------------

# ==============================
# START PROCESSING
# ==============================
print(f"Starting augmentation in {TRAIN_DIR}...")
for class_dir in TRAIN_DIR.iterdir():
    if class_dir.is_dir():
        print(f"\nProcessing class: {class_dir.name}")
        images = list(class_dir.glob("*.*"))  # jpg, jpeg, png
        for img_path in tqdm(images, desc=f"Augmenting {class_dir.name}"):
            if any(str(img_path).lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                # Resize original before augmentation
                image = cv2.resize(image, TARGET_SIZE)

                base_name = img_path.stem
                ext = img_path.suffix

                # Generate AUGMENT_COUNT images
                for i in range(AUGMENT_COUNT):
                    aug_img = image.copy()
                    aug_function = random.choice(AUG_FUNCTIONS)
                    aug_img = aug_function(aug_img)

                    # Save new file
                    new_filename = f"{base_name}_aug{i+1}{ext}"
                    cv2.imwrite(str(class_dir / new_filename), aug_img)

print("\n✅ Augmentation Completed Successfully!")
