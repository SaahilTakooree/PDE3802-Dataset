"""
YOLOv8 Classification Model - FIXED FOR OVERFITTING
Key changes: Stronger augmentation, regularization, smaller model, dropout
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import cv2

# ============================================
# CONFIGURATION - OPTIMIZED AGAINST OVERFITTING
# ============================================
DATASET_ROOT = Path("data")
PROJECT_NAME = "office_supplies_classifier"
MODEL_SIZE = "m"  # Changed from 'x' to 'm' - smaller model = less overfitting

CLASSES = [
    "erasers", "glue_sticks", "highlighters", "mugs", "paper_clips",
    "pencils", "pens", "staplers", "tapes", "usb_sticks"
]

# CRITICAL: Adjusted hyperparameters for generalization
EPOCHS = 100         # Increased - let early stopping decide when to stop
BATCH_SIZE = 32
IMG_SIZE = 224
PATIENCE = 5         # Reduced - stop sooner when validation plateaus
LEARNING_RATE = 0.0005  # Reduced for more stable training
DROPOUT = 0.3        # NEW: Add dropout for regularization

# ============================================
# CREATE YAML
# ============================================
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
    
    print(f"✓ Created data.yaml at {yaml_path}")
    return yaml_path

# ============================================
# TRAIN WITH ANTI-OVERFITTING MEASURES
# ============================================
def train_model():
    """Train with strong regularization to prevent overfitting"""
    
    yaml_path = create_data_yaml()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*50}")
    print(f"Training on: {device.upper()}")
    print(f"Model: YOLOv8{MODEL_SIZE}-cls (OPTIMIZED FOR GENERALIZATION)")
    print(f"{'='*50}\n")
    
    model = YOLO(f'yolov8{MODEL_SIZE}-cls.pt')
    
    # TRAIN WITH STRONGER REGULARIZATION
    results = model.train(
        data=str(DATASET_ROOT),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        patience=PATIENCE,  # Stop early when validation stops improving
        save=True,
        device=device,
        workers=8,
        project=PROJECT_NAME,
        name='train',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        
        # LEARNING RATE - Lower for stability
        lr0=LEARNING_RATE,      # Reduced initial LR
        lrf=0.001,              # Lower final LR
        momentum=0.937,
        weight_decay=0.001,     # INCREASED weight decay (L2 regularization)
        
        warmup_epochs=5,        # Longer warmup
        warmup_momentum=0.8,
        cos_lr=True,
        verbose=True,
        
        # CRITICAL: DISABLE CACHE to prevent memorization
        cache=False,            # Changed from True - forces reading fresh data
        
        amp=True,
        
        # STRONGER DATA AUGMENTATION (increased from your settings)
        hsv_h=0.03,            # Increased color variation
        hsv_s=0.7,             # Increased saturation changes
        hsv_v=0.4,             # Increased brightness changes
        degrees=25.0,          # Increased rotation
        translate=0.2,         # Increased translation
        scale=0.5,             # Increased scale variation
        shear=5.0,             # Added shear transformation
        perspective=0.0005,    # Added perspective transform
        flipud=0.0,
        fliplr=0.5,
        
        # ADDITIONAL AUGMENTATION
        mosaic=0.0,            # Keep 0 for classification
        mixup=0.15,            # ENABLED mixup - mixes training images
        copy_paste=0.0,
        
        # DROPOUT (if supported by your ultralytics version)
        dropout=DROPOUT,       # Adds dropout layers
        
        # LABEL SMOOTHING - Prevents overconfident predictions
        label_smoothing=0.1,   # NEW: Softens hard labels
    )
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print("\n⚠️  Check validation metrics vs training metrics:")
    print("   If validation accuracy is much lower, overfitting may still exist")
    
    return model

# ============================================
# VALIDATION
# ============================================
def validate_model(model_path=None):
    """Validate and compare train vs test performance"""
    
    if model_path is None:
        model_path = f'{PROJECT_NAME}/train/weights/best.pt'
    
    model = YOLO(model_path)
    
    # Validate on VALIDATION set
    print("\n📊 Validating on VALIDATION set...")
    val_metrics = model.val(
        data=str(DATASET_ROOT / 'data.yaml'),
        split='val'
    )
    
    # Validate on TEST set
    print("\n📊 Validating on TEST set...")
    test_metrics = model.val(
        data=str(DATASET_ROOT / 'data.yaml'),
        split='test'
    )
    
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    print(f"Validation Top-1: {val_metrics.top1:.2%}")
    print(f"Test Top-1:       {test_metrics.top1:.2%}")
    print(f"Validation Top-5: {val_metrics.top5:.2%}")
    print(f"Test Top-5:       {test_metrics.top5:.2%}")
    
    # Calculate overfitting gap
    gap = val_metrics.top1 - test_metrics.top1
    print(f"\n⚠️  Generalization Gap: {gap:.2%}")
    
    if abs(gap) < 0.05:
        print("✅ Good generalization!")
    elif abs(gap) < 0.10:
        print("⚠️  Moderate overfitting - consider more regularization")
    else:
        print("❌ Significant overfitting - increase augmentation/regularization")
    
    return val_metrics, test_metrics

# ============================================
# PREDICT WITH CONFIDENCE THRESHOLD
# ============================================
def predict_image(image_path, model_path=None, conf_threshold=0.4):
    """Predict with adjustable confidence threshold"""
    
    if model_path is None:
        model_path = f'{PROJECT_NAME}/train/weights/best.pt'
    
    model = YOLO(model_path)
    
    results = model.predict(
        source=image_path,
        imgsz=IMG_SIZE,
        conf=conf_threshold,  # Higher threshold for production
        verbose=False
    )
    
    result = results[0]
    top1_class = result.names[result.probs.top1]
    top1_conf = result.probs.top1conf.item()
    
    print(f"\n{'='*50}")
    print(f"Image: {Path(image_path).name}")
    print(f"{'='*50}")
    print(f"🎯 Prediction: {top1_class.upper()}")
    print(f"📊 Confidence: {top1_conf:.2%}")
    
    if top1_conf < 0.6:
        print("⚠️  Low confidence - model is uncertain")
    
    # Show top 5
    print("\nTop 5 Predictions:")
    print("-" * 50)
    for idx, conf in zip(result.probs.top5, result.probs.top5conf):
        print(f"  {result.names[idx]:15s} → {conf:.2%}")
    print("="*50)
    
    return top1_class, top1_conf

# ============================================
# ADDITIONAL: TEST TIME AUGMENTATION (TTA)
# ============================================
def predict_with_tta(image_path, model_path=None):
    """Use Test-Time Augmentation for more robust predictions"""
    
    if model_path is None:
        model_path = f'{PROJECT_NAME}/train/weights/best.pt'
    
    model = YOLO(model_path)
    
    # Predict with augmentation
    results = model.predict(
        source=image_path,
        imgsz=IMG_SIZE,
        augment=True,  # Enable TTA
        verbose=False
    )
    
    result = results[0]
    top1_class = result.names[result.probs.top1]
    top1_conf = result.probs.top1conf.item()
    
    print(f"\n🔬 Test-Time Augmentation Result:")
    print(f"   Prediction: {top1_class} ({top1_conf:.2%})")
    
    return top1_class, top1_conf


def predict_webcam(model_path=None, conf_threshold=0.4):
    """Real-time webcam classification"""
    if model_path is None:
        model_path = f'{PROJECT_NAME}/train/weights/best.pt'

    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)

    print("\n🎥 Starting webcam... Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Cannot access webcam.")
            break

        results = model.predict(source=frame, imgsz=IMG_SIZE, conf=conf_threshold, verbose=False)
        result = results[0]
        top1_class = result.names[result.probs.top1]
        top1_conf = result.probs.top1conf.item()

        # Display on frame
        label = f"{top1_class} ({top1_conf:.2%})"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.imshow("YOLOv8 Webcam Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8 Office Supplies - Anti-Overfitting')
    parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'validate', 'predict', 'predict-tta', 'webcam'],
                    help='Mode to run')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image for prediction')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("\n🚀 Training with anti-overfitting measures...")
        model = train_model()
        print("\n✅ Training finished!")
        print("\n📌 Next step: Run validation to check generalization")
        
    elif args.mode == 'validate':
        print("\n🔍 Validating model...")
        validate_model(args.model)
        
    elif args.mode == 'predict':
        if not args.image:
            print("❌ Error: --image required for prediction")
        else:
            predict_image(args.image, args.model)
            
    elif args.mode == 'predict-tta':
        if not args.image:
            print("❌ Error: --image required for prediction")
        else:
            predict_with_tta(args.image, args.model)
    
    elif args.mode == 'webcam':
        print("\n🎥 Running webcam live classification...")
        predict_webcam(args.model)
