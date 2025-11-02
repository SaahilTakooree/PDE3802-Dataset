# Office Item Classifier Dataset

## 1. Metadata Block

| **Field** | **Required?** | **Value** |
|------------|---------------|-----------|
| **Dataset Name** | ✅ | Office Item Classifier Dataset |
| **Version** | ✅ | v1.0 |
| **Authors / Maintainers** | ✅ | Teena Busgeeth, Kritisha Sunjhoreea, Meharsing Takooree |
| **Date Created** | ✅ | 2025-09-26 |
| **Last Updated** | ✅ | 2025-11-30 |
| **License** | ✅ | Pending — dataset created for academic coursework use only (no public redistribution). |
| **Language** | ✅ | English |
| **Homepage / Repository** | ✅ | [https://github.com/SaahilTakooree/PDE3802-Dataset](https://github.com/SaahilTakooree/PDE3802-Dataset) |
| **Citation / DOI** | ✅ | _No DOI assigned. Please cite as: **Busgeeth, T., Sunjhoreea, K., & Takooree, M. (2025). *Office Item Classifier Dataset (v1.0).* GitHub Repository.** |
| **Contact** | ✅ | TB852@live.mdx.ac.uk, KS1745@live.mdx.ac.uk, MT1213@live.mdx.ac.uk |

---

## 2. Summary of Dataset

The Office Item Classifier Dataset is hand-picked collection of 10,000 images. These images span 10 common office supply items. There are items like pens, pencils, staplers, and USB sticks among others. The pictures are taken under different conditions with changes in their position, light, and background. The images were also augmented to enhance the model robustness.

The dataset was created to address the problem of accurately differentiate between visually similar office items in real-world settings. For example, distinguishing between a pen from a pencil or a glue stick from a highlighter. This is relevant for automated inventory systems, and automated office desk sorting systems. While the dataset provides a rich training resources, the dataset is not to be used for decision-making outside of object classification, facial recognition, or personal data inference. Hence the reason why it does not contain people or personal context and it is only limited to office supply items.

---

## 3. Composition of Dataset

The Dataset is contains 10,000 images of 10 common office items. These images are intended for training a robust image classification models using transfer learning. This dataset is carefully balanced across classes. Only the training set is augmented to improve model robustness while keeping validation and test set untouched for a fair evaluation.

### Classes

The dataset contains the following categories:

| **#** | **Classes** |
|-------|-------------|
|1.     |erasers      |
|2.     |glue_sticks  |
|3.     |highlighters |
|4.     |mugs         |
|5.     |paper_clips  |
|6.     |pencils      |
|7.     |pens         |
|8.     |staplers     |
|9.     |tapes        |
|10.    |usb_sticks   |

#### Rational for class selection:
- **Fine-grained vs distinctive:** Some items in the dataset is visually similar like pens vs pencils. This was deliberately done to test for subtle feature learning. On the other hand, some items have distinctive shapes like mugs and staplers for easier baseline accuracy.
- **Size diversity:** The dataset composed of small object like paper clips and USB sticks to challenge detection at different scales.
- **Orientation variance:** Cylindrical items like glue sticks and highlighters to test for rotational invariance.
---

## 4. Collection Process.

Source(s): The dataset was compiled from a mixed set of sources, including publicly available datasets and images manually captured. Major sources include:
- Kaggle
- Roboflow Universe
- CV Images

Additional images captured manually with a digital camera in controlled lighting environments.

**Acquisition method:** Images were downloaded manually from the above sources. The manually captured images were taken by standard digital cameras at high resolution (12MP), then resized to the target resolution (224×224) during preprocessing.

**Filtering rules:**
- All duplicates were removed programmatically.
- Images that were not two clearly shown (blurred, partially hidden, or unmeaning) were eliminated.


**Verification:** All pictures got through the manual checking by various annotators to ensure the presence of the object and the right labeling. Automated scripts were not utilised past the initial stages of identifying duplicate files and sorting the files.

**Rationale:** This filtering method helps to ensure that the final set of images is characteristics of high quality, diverse, and correctly labeled, thus making it easier to reproduce the training and testing of classification models.

## 5. Preprocessing, Cleaning, and Augmentation

The entire set of images was subjected to preprocessing and augmentation in order to prepare a robust training dataset that is appropriate for YOLOv8 L CLS classification. The approach not only assures that the same outcomes are always attained, that there is consistency, and that the implications made are true to the actual situation.

### Data Preprocessing

- **Resizing:** The images, having a height and width of 224×224 pixels, were resized with OpenCV (cv2.resize) to be compatible with the input dimensions of YOLOv8 L CLS.

- **Normalisation:** To begin with, the pixel values of the model input were scaled down to the range [0, 1].

- **Deduplication and cleaning:** It was done by removing all the duplicate images, eliminating the objects that were out of focus or partially seen, and discarding the images that contained the branding or personal information.

- **Dataset Splitting:** The dataset was divided into stratified train (70%), validation (20%), and test (10%) sets with a fixed random seed (random.seed(100)) being used for reproducibility. The splits are composed of equal numbers of each class. Splitting was done by the use of sklearn.model_selection.train_test_split with the images being moved to the directory structure that is required by YOLOv8 (data/train/, data/val/, data/test/).

### Training Set Augmentation

The static (pre-generated images) and dynamic (on-the-fly during training) augmentations have both been added to the training set to make the model more robust and generalise better. Consequently, the model will be able to detect objects under any condition, be it orientation, size, brightness, background, partially hidden, and other situations that may occur in reality.

**1. Static (Pre-Generated) Augmentation:**

The original training dataset consists of 874 images per class, which are augmented four times giving a total of 3,495 images per class. The augmentation is done programmatically with OpenCV and NumPy, while the original images remain in the dataset. OpenCV and NumPy are used to perform the augmentation through a programmatic approach, and the original images are still present in the dataset. The filenames of the augmented images are different from the originals, and they are stored together (e.g., pen_aug1.jpg).

The following techniques are applied:

- **Random rotation:** ±15° to achieve invariance to rotation.

- **Random adjustment of brightness:** ±50 to mimic different lighting.

- **Random noise:** To represent the noise caused by the camera and the environment.

- **Horizontal flipping:** For symmetry of objects and reversed orientations.

**Justification:**

- Maintains distributions of real-world features and at the same time avoids overfitting.
- The balanced augmentation allows models to capture the real object features, not the artifacts.
- Computationally efficient: four variations per image yield diversity without consuming excessive training time.

**2. Dynamic (Training-Time) Augmentation:**

During the training process, more augmentations are dynamically applied to each batch to further enhance the diversity of the data. These augmentations are added onto the static augmentations and are regulated by training hyperparameters.

Techniques applied during training: 
- **Color augmentation (HSV):** Hue (hsv_h=0.8): slight shifts to tell apart objects that are very similar visually (e.g. pens vs. pencils).
- **Saturation (hsv_s=0.95):** Wide range of saturation variation.
- **Value (hsv_v=0.95):** Complete brightness variation (dim to bright).

Geometric / Multi-angle augmentation:
- **Rotation (degrees=270):** Up to ±270° to enhance rotation invariance.
- **Translation (translate=0.4):** Moderate shifts in the x/y directions.
- **Scale (scale=0.9):** Variations in zooming in/out.
- **Shear (shear=15):** Mimics the perspective deformation.
- **Perspective (perspective=0.002):** Very slight changes in perspective.
- Horizontal flip (fliplr=0.5) and vertical flip (flipud=0.5).

Regularisation / Advanced augmentation:
- **Mixup (mixup=0.5):** Two images are blended together to enhance the model's generalisation ability.
- **Random erasing (erasing=0.5):** Mimicking occlusions.
- **AutoAugment (auto_augment='randaugment'):** A systematic augmentation policy.
- **Random cropping (crop_fraction=0.55):** Teaching the model scale invariance.

Training settings impacting augmentation:
- Optimiser: AdamW.
- Learning rate: warmup + cosine decay (lr0, lrf).
- Regularisation: dropout and label smoothing (0.15).
- Mixed precision training enabled (amp=True).

Rationale:
- Each epoch introduces changes in the data, which in a way increases the diversity of the dataset without the necessity of additional images stored on the disk.
- The flexible modifications, together with the fixed augmentations, make the model harder to break and more capable of applying its knowledge practically in the case of such variations as rotation, scale, lighting, background noise, and occlusion.
- Renders the model incapable of adapting to the static augmentation artifacts and at the same time, maintaining its performance on the clean validation and test sets.

### Measures of Reproducibility

Used fixed random seeds throughout (random.seed(100)) for preprocessing and splitting.

Available scripts for the dataset splitting procedure (dataset_split.py) and static augmentation processing (train_augmentation.py).

Each augmented image gets a unique filename; the validation and testing datasets are left unchanged.

Each preprocessing step is thoroughly recorded so that the dataset can be independently recreated.

### Utilised Tools

Preprocessing and static augmentation: OpenCV, NumPy

Dataset splitting and file handling: Python pathlib, shutil, and scikit-learn

Dynamic augmentation: YOLOv8 L CLS built-in training transforms

Progress tracking made possible by tqdm

Result: The combination of preprocessing and augmentation results in a very diverse dataset that can be reproduced for YOLOv8 L CLS, thus enabling feature learning through fine-grained and recognition of office items.

---

## 6. Dataset Structure and Split Strategy

The dataset is structured in a class-folder manner, which is compatible with major deep learning frameworks like PyTorch (ImageFolder) and TensorFlow (image_dataset_from_directory) and so, manual labelling files are not necessary.

### Statistics of Dataset

| **Split** | **Images per Class** | **Total Images** |
|-----------|----------------------|------------------|
|Train      |3,495                 |34,950            |
|Validation |200                   |2,000             |
|Test       |100                   |1,000             |
|Total      |3795                  |37,950            |

#### Image Properties:
- Format: .jpg
- Size: 224 x 224 pixels
- Channels: RGB
- Labels: Inferred from folder names; No additonal data.

### Data Split Rationale
- **Training (70%):** There was 700 original images per class. Each original images was augmented 4 times. This resulted in 3,495 images per class. This was done to provides sufficient variety for deep learning learning without overfitting.
- **Validation (20%):** There was 200 images per class. This was large enough for hyperparameter tuning and early stopping.
- **Test(10%):** There was 100 images per class. These images was unseen holdout set to ensure unbiased perfomance evaluation.
- **Stratified splitting:** All splits maintain class balances.

### Directory Structure

The dataset is organised in a class-based folder structure, compatible with standard deep learning frameworks. The breakdown is as follows:

#### Training set (data/train/):

- erasers/ — 699 original images + 2,796 augmented images
- glue_sticks/ — 699 original + 2,796 augmented
- highlighters/ — 699 original + 2,796 augmented
- mugs/ — 699 original + 2,796 augmented
- paper_clips/ — 699 original + 2,796 augmented
- pencils/ — 699 original + 2,796 augmented
- pens/ — 699 original + 2,796 augmented
- staplers/ — 699 original + 2,796 augmented
- tapes/ — 699 original + 2,796 augmented
- usb_sticks/ — 699 original + 2,796 augmented

#### Validation set (data/val/):
- Each class folder contains 200 images, with no augmentation applied.

#### Test set (data/test/):
- Each class folder contains 100 images, with no augmentation applied.

##### Notes:
- Folder names correspond directly to class labels.
- Only the training set contains augmented images; validation and test sets are untouched to ensure fair evaluation.

## 7. Statistics and Visualisations
Dataset is balanced across classes with various object orientations, scales, and backgrounds.

### Class Distribution.

| **Class**   | **Train** | **Val** | **Test** |
|-----------  |-----------|---------|----------|
|glue_sticks  |3,495      |200      |100       |           
|highlighters |3,495      |200      |100       |
|mugs         |3,495      |200      |100       |
|paper_clips  |3,495      |200      |100       | 
|pencils      |3,495      |200      |100       |
|pens         |3,495      |200      |100       |
|staplers     |3,495      |200      |100       |
|tapes        |3,495      |200      |100       |
|usb_sticks   |3,495      |200      |100       |

#### Visualisations
- **Class distribution bar chart:** visually confirms class balance
- **Sample montage:** 3–5 images per class to show object variation
- **Basic statistics:** Mean pixel intensity per channel (R/G/B), standard deviation across all images

These statistics confirm the dataset is balanced, diverse, and ready for ML experiments.

--- 

## 8.Typical Usage Workflow

Ultralytics YOLOv8 has full compatibility with the dataset, and a typical training pipeline is delivered.

### Workflow Example:

YAML Configuration Generation

```python
def create_data_yaml():
    data_config = {
        'path': str(DATASET_ROOT.absolute()),
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'names': {i: name for i, name in enumerate(CLASSES)}
    }
    yaml_path = DATASET_ROOT / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, sort_keys=False)
````
This creates a data.yaml file for YOLOv8 L CLS specifying dataset paths and class labels.

### Train YOLOv8 L CLS Model

```python
from ultralytics import YOLO

model = YOLO('yolov8l-cls.pt')

results = model.train(
    data='data.yaml',
    epochs=EPOCHS,
    imgsz=224,
    batch=16,
    optimizer='AdamW',
    lr0=0.01,
    cos_lr=True,
    hsv_h=0.8,
    hsv_s=0.95,
    hsv_v=0.95,
    degrees=270,
    translate=0.4,
    scale=0.9,
    shear=15,
    perspective=0.002,
    fliplr=0.5,
    flipud=0.5,
    mixup=0.5,
    erasing=0.5,
    auto_augment='randaugment',
    crop_fraction=0.55,
    label_smoothing=0.15,
    dropout=DROPOUT,
    device='cuda',
    project='OfficeItemClassifier',
    name='Model_Office_Classifier'
)
```
### Expected Outputs
|Metric          |Definition                                      |Notes|
|----------------|------------------------------------------------|-----|
|Accuracy        |Overall % of correct predictions on the test set|	Measures general performance|
|Macro F1-score  |Class-balanced F1 across all 10 categories	  |Compensates for any residual imbalance|
|Confusion Matrix|10×10 grid showing class-wise misclassifications|	Used for error analysis (e.g., pen pencil confusions)|

These metrics are automatically computed and saved as runs/classify/Model_Office_Classifier_recheck/results.csv during the training of YOLOv8.

### Downstream / Extended Uses
|Use Case                  |Description|
|--------------------------|-------------|
|Automated Office Inventory|	Detect and classify items on desks for stock management|
|Smart Desk Systems        |	Integrate into AI-powered organisers that can visually recognise misplaced items|
|Benchmark Dataset for Fine-Grained Object Recognition|	Useful for evaluating transfer learning on visually similar classes|

### Ethical and Responsible Use

- **Intended Use:** Exclusively for purposes of education, academic research, and evaluations.

- **Prohibited Use:** No application related to personal identification, monitoring, or selling data.

---

## 9. Limitations

Despite the fact that the dataset was strong enough for the detection of office supplies, users should nevertheless take the limitation of each class into account: 

- **Glue Stick, Stapler, Tape:**

The detection is extremely reliable and can be done even on any kind of background (busy or plain), under various lighting conditions, inside or outside of the bags, and from different angles.

There are distinctive features between the objects so a clear focus will lead to the correct detection of object.


**Pen, Pencil, Highlighter:**

Detection is quite reliable provided that the tip of the item is sharp and focused.

Limitation: Blurred or partially occluded tips severely compromise detection accuracy.

a) Shape and surface texture:
A pen typically has  a smooth, glossy surface while a pencil has a matte or wooden texture with hexagonal or round body and sometimes an eraser and the top end. The Highlighter has a thicker and broader body with vivid fluorescent colour.

b) Tip characteristics:
A pen has a metallic nib, pencil has a tapered graphite tip and highlighter is recognised by its chisel-shaped tip.



**Mug:**

Detection may sometimes need to see the handle.

Limitation: Lack of visibility of the handle in the image may cause misclassification as a glass or detection failure.


**Paper Clips:**

The items become evident through their proximity to the central area of the frame and the degree of sharpness in focus.

Limitation: Clips that are a bit off-center or have not achieved a full focus might be left out.

**Eraser:**

If it is on a busy background, it has to be very near to the camera.

Limitation: Being far away or a very busy background can cause non-detection. The uniform matte surface with geometric form and colour consistency helps the camera to distinguish between eraser and USB stick

**USB Stick:**

On a busy background, detection is possible if it is close to the camera.

Limitation: The best results are on a plain background; busy backgrounds can depress reliability. The black or silver inner port (reflective metallic surface) of the USB makes the camera distinguish between a USB stick and eraser.
