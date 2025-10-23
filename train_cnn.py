"""
YOLOv8 Classification Model for Office Supplies
Classifies 10 objects: erasers, glue sticks, highlighters, mugs, paper clips, 
pencils, pens, staplers, tapes, usb sticks
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import cv2
import numpy as np
from PIL import Image

# ============================================
# CONFIGURATION
# ============================================
# Your data structure should be:
# dataset/
#   ├── train/
#   │   ├── erasers/
#   │   ├── glue_sticks/
#   │   ├── highlighters/
#   │   └── ...
#   ├── val/
#   │   ├── erasers/
#   │   └── ...
#   └── test/
#       ├── erasers/
#       └── ...

DATASET_ROOT = Path("data")  # Change this to your dataset path
PROJECT_NAME = "office_supplies_classifier"
MODEL_SIZE = "x"  # Options: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
                  # 'n' is fastest, 'x' is most accurate

CLASSES = [
    "erasers",
    "glue_sticks", 
    "highlighters",
    "mugs",
    "paper_clips",
    "pencils",
    "pens",
    "staplers",
    "tapes",
    "usb_sticks"
]

# Training hyperparameters
EPOCHS = 50      # Reduced - with 35k images you don't need as many epochs
BATCH_SIZE = 32  # Increased for large dataset (adjust based on GPU: 16/32/64)
IMG_SIZE = 224   # YOLOv8-cls uses 224x224 by default
PATIENCE = 10    # Early stopping patience (reduced for large dataset)
LEARNING_RATE = 0.001  # Initial learning rate

# ============================================
# STEP 1: CREATE YAML CONFIGURATION
# ============================================
def create_data_yaml():
    """Creates the data.yaml file for YOLOv8 classification"""
    
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
    
    print(f"✓ Created data.yaml at {yaml_path}")
    return yaml_path

# ============================================
# STEP 2: TRAIN THE MODEL
# ============================================
def train_model():
    """Train YOLOv8 classification model"""
    
    # Create data.yaml
    yaml_path = create_data_yaml()
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*50}")
    print(f"Training on: {device.upper()}")
    print(f"{'='*50}\n")
    
    # Load pretrained YOLOv8 classification model
    model = YOLO(f'yolov8{MODEL_SIZE}-cls.pt')  # e.g., yolov8n-cls.pt
    
    print(f"Model: YOLOv8{MODEL_SIZE}-cls")
    print(f"Classes: {len(CLASSES)}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}\n")
    
    # Train the model
    results = model.train(
        data=str(DATASET_ROOT),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        save=True,
        device=device,
        workers=8,        # Increased workers for large dataset
        project=PROJECT_NAME,
        name='train',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',  # AdamW works well with large datasets
        lr0=LEARNING_RATE,  # Initial learning rate
        lrf=0.01,          # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,   # Warmup for stable training
        warmup_momentum=0.8,
        cos_lr=True,       # Cosine learning rate scheduler
        verbose=True,
        cache=True,        # Cache images in RAM for faster training (if you have enough RAM)
        amp=True,        # Automatic Mixed Precision for faster training
        
        # Data augmentation (reduced for large dataset - you already have variety)
        hsv_h=0.015,      # HSV-Hue augmentation
        hsv_s=0.5,        # HSV-Saturation (reduced)
        hsv_v=0.3,        # HSV-Value (reduced)
        degrees=10.0,     # Rotation (+/- deg) (reduced)
        translate=0.1,    # Translation (+/- fraction)
        scale=0.3,        # Scaling (+/- gain) (reduced)
        shear=0.0,        # Shear (+/- deg)
        perspective=0.0,  # Perspective (+/- fraction)
        flipud=0.0,       # Flip up-down probability
        fliplr=0.5,       # Flip left-right probability
        mosaic=0.0,       # Mosaic augmentation (set to 0 for classification)
        mixup=0.0,        # Mixup augmentation
        copy_paste=0.0,   # Copy-paste augmentation
    )
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    
    return model

# ============================================
# STEP 3: VALIDATE THE MODEL
# ============================================
def validate_model(model_path=None):
    """Validate the trained model on test set"""
    
    if model_path is None:
        model_path = f'{PROJECT_NAME}/train/weights/best.pt'
    
    model = YOLO(model_path)
    
    # Validate on test set
    print("\nValidating on test set...")
    metrics = model.val(
        data=str(DATASET_ROOT / 'data.yaml'),
        split='test'
    )
    
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    print(f"Top-1 Accuracy: {metrics.top1:.2%}")
    print(f"Top-5 Accuracy: {metrics.top5:.2%}")
    
    return metrics

# ============================================
# STEP 4: INFERENCE/PREDICTION
# ============================================
def predict_image(image_path, model_path=None):
    """Predict single image"""
    
    if model_path is None:
        model_path = f'{PROJECT_NAME}/train/weights/best.pt'
    
    model = YOLO(model_path)
    
    # Run prediction
    results = model.predict(
        source=image_path,
        imgsz=IMG_SIZE,
        conf=0.25,  # Confidence threshold
        verbose=False
    )
    
    # Get results
    result = results[0]
    top1_class = result.names[result.probs.top1]
    top1_conf = result.probs.top1conf.item()
    
    # Get top 5 predictions
    top5_indices = result.probs.top5
    top5_conf = result.probs.top5conf
    
    print(f"\n{'='*50}")
    print(f"Image: {Path(image_path).name}")
    print(f"{'='*50}")
    print(f"🎯 Prediction: {top1_class.upper()}")
    print(f"📊 Confidence: {top1_conf:.2%}\n")
    
    print("Top 5 Predictions:")
    print("-" * 50)
    for idx, conf in zip(top5_indices, top5_conf):
        class_name = result.names[idx]
        print(f"  {class_name:15s} → {conf:.2%}")
    print("="*50)
    
    return top1_class, top1_conf

# ============================================
# STEP 5: WEBCAM REAL-TIME PREDICTION
# ============================================
def predict_webcam(model_path=None):
    """Real-time prediction from webcam"""
    
    if model_path is None:
        model_path = f'{PROJECT_NAME}/train/weights/best.pt'
    
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\n" + "="*50)
    print("WEBCAM CLASSIFICATION")
    print("="*50)
    print("Press 'q' to quit")
    print("="*50 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run prediction
        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=0.25,
            verbose=False
        )
        
        result = results[0]
        top1_class = result.names[result.probs.top1]
        top1_conf = result.probs.top1conf.item()
        
        # Display on frame
        cv2.putText(
            frame,
            f"{top1_class}: {top1_conf:.2%}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )
        
        # Show top 3
        top3_indices = result.probs.top5[:3]
        top3_conf = result.probs.top5conf[:3]
        
        y_pos = 100
        for idx, conf in zip(top3_indices, top3_conf):
            class_name = result.names[idx]
            cv2.putText(
                frame,
                f"{class_name}: {conf:.1%}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            y_pos += 40
        
        cv2.imshow('Office Supplies Classifier', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ============================================
# STEP 6: EXPORT MODEL (Optional)
# ============================================
def export_model(model_path=None, format='onnx'):
    """Export model to different formats"""
    
    if model_path is None:
        model_path = f'{PROJECT_NAME}/train/weights/best.pt'
    
    model = YOLO(model_path)
    
    # Export (options: 'onnx', 'torchscript', 'tensorflow', 'tflite', etc.)
    model.export(format=format)
    print(f"✓ Model exported to {format.upper()} format")

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8 Office Supplies Classifier')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'validate', 'predict', 'webcam', 'export'],
                       help='Mode to run: train, validate, predict, webcam, or export')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image for prediction')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (default: best.pt)')
    parser.add_argument('--export-format', type=str, default='onnx',
                       help='Export format (onnx, torchscript, etc.)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("\n🚀 Starting Training...")
        model = train_model()
        print("\n✅ Training finished! Model saved.")
        
    elif args.mode == 'validate':
        print("\n📊 Running Validation...")
        validate_model(args.model)
        
    elif args.mode == 'predict':
        if args.image is None:
            print("❌ Error: Please provide --image path")
        else:
            predict_image(args.image, args.model)
            
    elif args.mode == 'webcam':
        print("\n📹 Starting Webcam Prediction...")
        predict_webcam(args.model)
        
    elif args.mode == 'export':
        print(f"\n📦 Exporting model to {args.export_format.upper()}...")
        export_model(args.model, args.export_format)

    print("\n✨ Done!\n")