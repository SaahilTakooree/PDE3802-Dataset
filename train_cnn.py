"""
YOLOv8 Classification - OPTIMIZED FOR SIMILAR-LOOKING OBJECTS
Focus: Better feature learning for pens/pencils/highlighters distinction
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import cv2

# ============================================
# CONFIGURATION - OPTIMIZED FOR FINE-GRAINED CLASSIFICATION
# ============================================
DATASET_ROOT = Path("data")
PROJECT_NAME = "office_supplies_classifier"

# KEY CHANGE: Use larger model for better feature learning
MODEL_SIZE = "l"  # Changed to 'l' (large) - better at subtle differences

CLASSES = [
    "erasers", "glue_sticks", "highlighters", "mugs", "paper_clips",
    "pencils", "pens", "staplers", "tapes", "usb_sticks"
]

# CRITICAL: Adjusted for better convergence
EPOCHS = 100
BATCH_SIZE = 16          # Reduced for larger model + better gradient stability
IMG_SIZE = 320           # INCREASED from 224 - captures more detail
PATIENCE = 15            # INCREASED - allow more time to learn
LEARNING_RATE = 0.0001   # LOWER - more careful learning for subtle features
DROPOUT = 0.2            # Reduced slightly

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
# TRAIN - OPTIMIZED FOR FINE-GRAINED DIFFERENCES
# ============================================
def train_model():
    """Train with focus on learning subtle object differences"""
    
    yaml_path = create_data_yaml()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"🎯 FINE-GRAINED CLASSIFICATION TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device.upper()}")
    print(f"Model: YOLOv8{MODEL_SIZE}-cls (Better feature extraction)")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE} (Higher resolution)")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Patience: {PATIENCE} epochs")
    print(f"{'='*60}\n")
    
    model = YOLO(f'yolov8{MODEL_SIZE}-cls.pt')
    
    # OPTIMIZED TRAINING SETTINGS
    results = model.train(
        data=str(DATASET_ROOT),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,           # Higher resolution for detail
        patience=PATIENCE,        # More patience for convergence
        save=True,
        save_period=10,           # Save every 10 epochs
        device=device,
        workers=4,                # Reduced workers for stability
        project=PROJECT_NAME,
        name='train_v2',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        
        # LEARNING RATE - Very conservative for fine details
        lr0=LEARNING_RATE,        # Lower initial LR
        lrf=0.0001,               # Very low final LR
        momentum=0.9,             # Slightly lower momentum
        weight_decay=0.0005,      # Moderate regularization
        
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,
        cos_lr=True,              # Cosine annealing
        
        verbose=True,
        cache=False,              # No caching
        amp=True,                 # Mixed precision
        
        # MODERATE AUGMENTATION - Don't destroy fine details
        hsv_h=0.02,               # Reduced - preserve color info
        hsv_s=0.5,                # Moderate saturation
        hsv_v=0.3,                # Moderate brightness
        degrees=15.0,             # Reduced rotation - preserve orientation
        translate=0.15,           # Moderate translation
        scale=0.4,                # Moderate scaling
        shear=2.0,                # Minimal shear
        perspective=0.0002,       # Minimal perspective
        flipud=0.0,               # No vertical flip
        fliplr=0.3,               # Less horizontal flip
        
        # ADVANCED AUGMENTATION
        mosaic=0.0,               # Off for classification
        mixup=0.1,                # Reduced - don't confuse similar objects
        copy_paste=0.0,
        auto_augment='randaugment',  # Use RandAugment
        erasing=0.0,              # No random erasing
        crop_fraction=1.0,        # Keep full image
        
        # REGULARIZATION
        dropout=DROPOUT,
        label_smoothing=0.05,     # Reduced - we want confident predictions
        
        # TRAINING DYNAMICS
        close_mosaic=10,          # Disable mosaic in last 10 epochs
        plots=True,               # Generate plots
        val=True,                 # Validate during training
    )
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📁 Model saved to: {PROJECT_NAME}/train_v2/weights/best.pt")
    print(f"📊 Results saved to: {PROJECT_NAME}/train_v2/")
    
    return model

# ============================================
# VALIDATION WITH DETAILED ANALYSIS
# ============================================
def validate_model(model_path=None):
    """Validate with per-class breakdown"""
    
    if model_path is None:
        model_path = f'{PROJECT_NAME}/train_v2/weights/best.pt'
    
    model = YOLO(model_path)
    
    print("\n📊 Validating on VALIDATION set...")
    val_metrics = model.val(
        data=str(DATASET_ROOT / 'data.yaml'),
        split='val',
        plots=True  # Generate confusion matrix
    )
    
    print("\n📊 Validating on TEST set...")
    test_metrics = model.val(
        data=str(DATASET_ROOT / 'data.yaml'),
        split='test',
        plots=True
    )
    
    print("\n" + "="*60)
    print("📈 PERFORMANCE COMPARISON")
    print("="*60)
    print(f"Validation Top-1: {val_metrics.top1:.2%}")
    print(f"Test Top-1:       {test_metrics.top1:.2%}")
    print(f"Validation Top-5: {val_metrics.top5:.2%}")
    print(f"Test Top-5:       {test_metrics.top5:.2%}")
    
    gap = val_metrics.top1 - test_metrics.top1
    print(f"\n⚠️  Generalization Gap: {gap:.2%}")
    
    if abs(gap) < 0.05:
        print("✅ Excellent generalization!")
    elif abs(gap) < 0.10:
        print("⚠️  Good, but could improve")
    else:
        print("❌ Needs improvement")
    
    print("\n💡 Check the confusion matrix at:")
    print(f"   {PROJECT_NAME}/train_v2/confusion_matrix.png")
    
    return val_metrics, test_metrics

# ============================================
# PREDICT WITH ENSEMBLE (MULTIPLE INFERENCES)
# ============================================
def predict_image(image_path, model_path=None, conf_threshold=0.3):
    """Predict with lower threshold to see uncertainty"""
    
    if model_path is None:
        model_path = f'{PROJECT_NAME}/train_v2/weights/best.pt'
    
    model = YOLO(model_path)
    
    # Run prediction
    results = model.predict(
        source=image_path,
        imgsz=IMG_SIZE,
        conf=conf_threshold,
        verbose=False
    )
    
    result = results[0]
    top1_class = result.names[result.probs.top1]
    top1_conf = result.probs.top1conf.item()
    
    print(f"\n{'='*60}")
    print(f"📸 Image: {Path(image_path).name}")
    print(f"{'='*60}")
    print(f"🎯 Prediction: {top1_class.upper().replace('_', ' ')}")
    print(f"📊 Confidence: {top1_conf:.2%}")
    
    # Enhanced warnings based on confidence
    if top1_conf < 0.5:
        print("⚠️  LOW CONFIDENCE - Model is very uncertain!")
        print("💡 Tip: The image might be ambiguous or outside training distribution")
    elif top1_conf < 0.7:
        print("⚠️  Moderate confidence - Consider the top 3 predictions")
    elif top1_conf > 0.95:
        print("✅ Very high confidence - Likely correct")
        # But warn if TOO confident (might be wrong)
        if top1_conf > 0.99:
            print("📌 Note: Extremely high confidence - verify if prediction seems wrong")
    else:
        print("✅ High confidence")
    
    # Show top 5 with gap analysis
    print("\n📊 Top 5 Predictions:")
    print("-" * 60)
    probs = list(zip(result.probs.top5, result.probs.top5conf))
    for i, (idx, conf) in enumerate(probs, 1):
        class_name = result.names[idx].replace('_', ' ')
        bar = "█" * int(conf * 50)
        print(f"  {i}. {class_name:15s} {conf:6.2%} {bar}")
    
    # Show confidence gap
    if len(probs) >= 2:
        gap = probs[0][1].item() - probs[1][1].item()
        print(f"\n💡 Confidence gap (1st vs 2nd): {gap:.2%}")
        if gap < 0.1:
            print("   ⚠️  Close call between top predictions!")
            print("   💡 Consider both possibilities or use TTA mode")
        elif gap < 0.3:
            print("   ⚠️  Model somewhat uncertain - check top 2-3 predictions")
    
    # Warning for common confusions
    confusion_pairs = [
        ('pens', 'pencils'), ('pencils', 'pens'),
        ('highlighters', 'pens'), ('highlighters', 'pencils'),
        ('erasers', 'paper_clips'), ('paper_clips', 'erasers')
    ]
    
    if len(probs) >= 2:
        top1 = result.names[probs[0][0]]
        top2 = result.names[probs[1][0]]
        if (top1, top2) in confusion_pairs:
            print(f"   ⚠️  Common confusion: {top1} ↔ {top2}")
            print(f"   💡 Try 'predict-tta' mode for better accuracy")
    
    print("="*60)
    
    return top1_class, top1_conf

# ============================================
# PREDICT WITH TEST-TIME AUGMENTATION
# ============================================
def predict_with_tta(image_path, model_path=None):
    """Use TTA for more robust predictions"""
    
    if model_path is None:
        model_path = f'{PROJECT_NAME}/train_v2/weights/best.pt'
    
    model = YOLO(model_path)
    
    print("\n🔬 Running Test-Time Augmentation...")
    
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
    
    print(f"\n{'='*60}")
    print(f"🔬 TTA Result: {top1_class.upper().replace('_', ' ')}")
    print(f"📊 Confidence: {top1_conf:.2%}")
    print(f"{'='*60}")
    
    return top1_class, top1_conf

# ============================================
# WEBCAM WITH IMPROVED UI
# ============================================
def predict_webcam(model_path=None, camera_id=0):
    """Real-time webcam classification"""
    
    if model_path is None:
        model_path = f'{PROJECT_NAME}/train_v2/weights/best.pt'
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n" + "="*60)
    print("📷 WEBCAM CLASSIFICATION")
    print("="*60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save frame")
    print("  'f' - Toggle FPS")
    print("="*60 + "\n")
    
    show_fps = True
    frame_count = 0
    import time
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run prediction
        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=0.2,  # Lower threshold to see uncertainty
            verbose=False
        )
        
        result = results[0]
        top1_class = result.names[result.probs.top1]
        top1_conf = result.probs.top1conf.item()
        
        # Create dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (700, 280), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Color based on confidence
        if top1_conf > 0.7:
            color = (0, 255, 0)  # Green
            status = "HIGH CONFIDENCE"
        elif top1_conf > 0.5:
            color = (0, 255, 255)  # Yellow
            status = "MODERATE"
        else:
            color = (0, 165, 255)  # Orange
            status = "LOW - UNCERTAIN"
        
        # Main prediction
        cv2.putText(frame, f"{top1_class.upper().replace('_', ' ')}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame, f"{top1_conf:.1%} - {status}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Confidence bar
        bar_width = int(600 * top1_conf)
        cv2.rectangle(frame, (10, 100), (610, 125), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 100), (10 + bar_width, 125), color, -1)
        
        # Top 3
        cv2.putText(frame, "Top 3:", (10, 155), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        y_pos = 185
        for i, (idx, conf) in enumerate(zip(result.probs.top5[:3], 
                                            result.probs.top5conf[:3])):
            name = result.names[idx].replace('_', ' ')
            cv2.putText(frame, f"{i+1}. {name}: {conf:.1%}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 
                       (255, 255, 255), 1)
            y_pos += 30
        
        # FPS
        if show_fps:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", 
                       (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('YOLOv8 Office Supplies - Press Q to quit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved: {filename}")
        elif key == ord('f'):
            show_fps = not show_fps
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Webcam closed")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8 Fine-Grained Classification')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'validate', 'predict', 'predict-tta', 'webcam'],
                       help='Mode to run')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image for prediction')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("\n🚀 Starting fine-grained classification training...")
        print("💡 This will take longer but learn better features")
        model = train_model()
        print("\n✅ Training complete!")
        print("📌 Next: Run validation to check improvements")
        
    elif args.mode == 'validate':
        validate_model(args.model)
        
    elif args.mode == 'predict':
        if not args.image:
            print("❌ Error: --image required")
        else:
            predict_image(args.image, args.model)
            
    elif args.mode == 'predict-tta':
        if not args.image:
            print("❌ Error: --image required")
        else:
            predict_with_tta(args.image, args.model)
    
    elif args.mode == 'webcam':
        predict_webcam(args.model, args.camera)