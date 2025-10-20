import shutil, random
from pathlib import Path
from sklearn.model_selection import train_test_split


random.seed(100)


source = Path("dataset")
destination = Path("data")


train_ratio = 0.7
val_ratio = 0.2


test_ratio = 0.1
abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6


if destination.exists():
    print("WARNING: destination exists. Remove/backup if you want fresh split.")
destination.mkdir(parents=True, exist_ok=True)


def make_dirs(split, classes):
    for c in classes:
        (destination / split / c).mkdir(parents=True, exist_ok=True)


folders = []
classes = []

all_items = source.iterdir()

for item in all_items:
    if item.is_dir():
        folders.append(item)

for folder in folders:
    classes.append(folder.name)
    
    
make_dirs("train", classes)
make_dirs("val", classes)
make_dirs("test", classes)

for img_class in classes:
    imgs = []
    
    images = list((source/img_class).glob("*"))
    for p in images:
        imgs.append(str(p))

    train_and_val, test = train_test_split(imgs, test_size=test_ratio, random_state=100, stratify=None)
    val_size = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(train_and_val, test_size=val_size, random_state=100)

    splits = {
            "train": train,
            "val": val,
            "test": test
        }
    for split_name, image_list in splits.items():
        for image_path in image_list:
            image_name = Path(image_path).name
            
            dest_folder = destination / split_name / img_class
            
            dest_path = dest_folder / image_name
            
            shutil.copy(image_path, dest_path)
