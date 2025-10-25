"""
YOLOv8 Classification - FINAL VERSION
Fixes: Background interference + Pen/Pencil/Highlighter confusion
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import cv2
import numpy as np

# ============================================
# CONFIGURATION - ANTI-BACKGROUND + SHAPE FOCUS
# ============================================
DATASET_ROOT = Path("data")
PROJECT_NAME = "office_supplies_classifier"
MODEL_SIZE = "l"

CLASSES = [
    "erasers", "glue_sticks", "highlighters", "mugs", "paper_clips",
    "pencils", "pens", "staplers", "tapes", "usb_sticks"
]

# OPTIMIZED FOR: Ignore backgrounds + Distinguish similar objects
EPOCHS = 150
BATCH_SIZE = 16
IMG_SIZE = 384               # INCREASED - better detail for pen tips
PATIENCE = 30                # Very high - noisy training
LEARNING_RATE = 0.00003      # Very low for careful learning
DROPOUT = 0.4                # High dropout

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
# TRAIN - ANTI-BACKGROUND + SHAPE FOCUS
# ============================================
def train_model():
    """Train to ignore backgrounds and focus on object shapes"""
    
    yaml_path = create_data_yaml()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"🎯 ANTI-BACKGROUND + SHAPE-FOCUSED TRAINING")
    print(f"{'='*70}")
    print(f"Issue #1 Fix: Heavy crops + erasing to ignore backgrounds")
    print(f"Issue #2 Fix: High resolution + extreme aug for pen/pencil/highlighter")
    print(f"{'='*70}")
    print(f"Device: {device.upper()}")
    print(f"Model: YOLOv8{MODEL_SIZE}-cls")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE} (HIGH for detail)")
    print(f"Strategy: Force attention to CENTER object only")
    print(f"{'='*70}\n")
    
    model = YOLO(f'yolov8{MODEL_SIZE}-cls.pt')
    
    results = model.train(
        data=str(DATASET_ROOT),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,          # Higher for pen tip details
        patience=PATIENCE,
        save=True,
        save_period=10,
        device=device,
        workers=4,
        project=PROJECT_NAME,
        name='train_v4_final',   # New version
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        
        # LEARNING RATE
        lr0=LEARNING_RATE,
        lrf=0.00001,
        momentum=0.937,
        weight_decay=0.001,      # Increased regularization
        
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,
        cos_lr=True,
        
        verbose=True,
        cache=False,
        amp=True,
        
        # EXTREME COLOR AUGMENTATION - Ignore colors completely
        hsv_h=0.8,               # MAXIMUM - all colors
        hsv_s=0.95,              # MAXIMUM saturation
        hsv_v=0.95,              # MAXIMUM brightness
        
        # CRITICAL: HEAVY CROPS - Forces focus on CENTER object
        degrees=180.0,           # All rotations
        translate=0.4,           # INCREASED - move object around
        scale=0.95,              # INCREASED - zoom in/out heavily
        shear=15.0,              # Heavy distortion
        perspective=0.002,       # Perspective changes
        flipud=0.5,
        fliplr=0.5,
        
        # BACKGROUND ELIMINATION STRATEGY
        mosaic=0.0,
        mixup=0.5,               # INCREASED - blend with other images (confuses background)
        copy_paste=0.0,
        auto_augment='randaugment',
        erasing=0.6,             # CRITICAL: Randomly erase 60% of patches (forces partial recognition)
        crop_fraction=0.5,       # CRITICAL: Only see 50% of image (ignore edges/background)
        
        # REGULARIZATION
        dropout=DROPOUT,
        label_smoothing=0.15,    # Higher - less overconfident
        
        # TRAINING DYNAMICS
        close_mosaic=15,
        plots=True,
        val=True,
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"📁 Model: {PROJECT_NAME}/train_v4_final/weights/best.pt")
    print(f"💡 Test with backgrounds - should ignore them now!")
    print(f"💡 Test pen/pencil/highlighter - should distinguish better!")
    
    return model

# ============================================
# VALIDATION
# ============================================
def validate_model(model_path=None):
    """Validate model"""
    
    if model_path is None:
        # Auto-select best available
        v4_path = f'{PROJECT_NAME}/train_v4_final/weights/best.pt'
        v3_path = f'{PROJECT_NAME}/train_v3_extreme_aug/weights/best.pt'
        v2_path = f'{PROJECT_NAME}/train_v2/weights/best.pt'
        
        if Path(v4_path).exists():
            model_path = v4_path
        elif Path(v3_path).exists():
            model_path = v3_path
        else:
            model_path = v2_path
    
    model = YOLO(model_path)
    
    print(f"\n💡 Using model: {model_path}\n")
    
    print("📊 Validating on VALIDATION set...")
    val_metrics = model.val(
        data=str(DATASET_ROOT / 'data.yaml'),
        split='val',
        plots=True
    )
    
    print("\n📊 Validating on TEST set...")
    test_metrics = model.val(
        data=str(DATASET_ROOT / 'data.yaml'),
        split='test',
        plots=True
    )
    
    print("\n" + "="*60)
    print("📈 RESULTS")
    print("="*60)
    print(f"Validation: {val_metrics.top1:.2%}")
    print(f"Test:       {test_metrics.top1:.2%}")
    print(f"Gap:        {(val_metrics.top1 - test_metrics.top1):.2%}")
    
    if abs(val_metrics.top1 - test_metrics.top1) < 0.05:
        print("✅ Excellent generalization!")
    
    return val_metrics, test_metrics

# ============================================
# PREDICT WITH CENTER-CROP PREPROCESSING
# ============================================
def preprocess_image_center_focus(image_path):
    """Preprocess: Focus on center, blur background"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Create center mask (focus on middle 60%)
    mask = np.zeros((h, w), dtype=np.uint8)
    center_h, center_w = int(h * 0.6), int(w * 0.6)
    start_h, start_w = (h - center_h) // 2, (w - center_w) // 2
    mask[start_h:start_h+center_h, start_w:start_w+center_w] = 255
    
    # Blur background slightly
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    
    # Combine: sharp center, blurred edges
    result = np.where(mask[:,:,None] == 255, img, blurred)
    
    return result.astype(np.uint8)

def predict_image(image_path, model_path=None, conf_threshold=0.3, use_preprocessing=False):
    """Predict with optional center-focus preprocessing"""
    
    if model_path is None:
        v4_path = f'{PROJECT_NAME}/train_v4_final/weights/best.pt'
        v3_path = f'{PROJECT_NAME}/train_v3_extreme_aug/weights/best.pt'
        v2_path = f'{PROJECT_NAME}/train_v2/weights/best.pt'
        
        if Path(v4_path).exists():
            model_path = v4_path
        elif Path(v3_path).exists():
            model_path = v3_path
        else:
            model_path = v2_path
    
    model = YOLO(model_path)
    
    # Optional preprocessing
    source = image_path
    if use_preprocessing:
        preprocessed = preprocess_image_center_focus(image_path)
        if preprocessed is not None:
            source = preprocessed
            print("💡 Applied center-focus preprocessing")
    
    # Run prediction
    results = model.predict(
        source=source,
        imgsz=IMG_SIZE,
        conf=conf_threshold,
        verbose=False
    )
    
    result = results[0]
    top1_class = result.names[result.probs.top1]
    top1_conf = result.probs.top1conf.item()
    
    print(f"\n{'='*60}")
    print(f"📸 Image: {Path(image_path).name}")
    print(f"🤖 Model: {Path(model_path).stem}")
    print(f"{'='*60}")
    print(f"🎯 Prediction: {top1_class.upper().replace('_', ' ')}")
    print(f"📊 Confidence: {top1_conf:.2%}")
    
    # Confidence interpretation
    probs = list(zip(result.probs.top5, result.probs.top5conf))
    
    if top1_conf < 0.5:
        print("⚠️  LOW CONFIDENCE - Very uncertain")
        print("💡 Tip: Try better lighting or clearer angle")
    elif top1_conf < 0.7:
        print("⚠️  MODERATE - Check top 3")
    elif top1_conf > 0.95 and len(probs) >= 2 and probs[1][1].item() < 0.02:
        print("⚠️  VERY HIGH CONFIDENCE (check if correct!)")
        print("💡 Overconfidence sometimes indicates unusual input")
    else:
        print("✅ Good confidence")
    
    # Show top 5
    print("\n📊 Top 5 Predictions:")
    print("-" * 60)
    for i, (idx, conf) in enumerate(probs, 1):
        class_name = result.names[idx].replace('_', ' ')
        bar = "█" * int(conf * 50)
        print(f"  {i}. {class_name:15s} {conf:6.2%} {bar}")
    
    # Confidence gap analysis
    if len(probs) >= 2:
        gap = probs[0][1].item() - probs[1][1].item()
        print(f"\n💡 Gap (1st vs 2nd): {gap:.2%}")
        
        if gap < 0.15:
            print("   ⚠️  Close call! Consider both options")
            
            # Specific advice for pen/pencil/highlighter
            top1_name = result.names[probs[0][0]]
            top2_name = result.names[probs[1][0]]
            
            if {top1_name, top2_name} <= {'pens', 'pencils', 'highlighters'}:
                print("   🔍 PEN/PENCIL/HIGHLIGHTER confusion detected!")
                print("   💡 Tips to distinguish:")
                print("      - Pens: Usually have clips/caps, smooth body")
                print("      - Pencils: Hexagonal, wood texture, graphite tip")
                print("      - Highlighters: Wider tip, translucent body")
    
    print("="*60)
    
    return top1_class, top1_conf

# ============================================
# WEBCAM WITH BACKGROUND BLUR OPTION
# ============================================
def predict_webcam(model_path=None, camera_id=0, blur_background=False):
    """Real-time webcam with optional background blur + center guide + high FPS"""

    if model_path is None:
        v4_path = f'{PROJECT_NAME}/train_v4_final/weights/best.pt'
        v3_path = f'{PROJECT_NAME}/train_v3_extreme_aug/weights/best.pt'
        v2_path = f'{PROJECT_NAME}/train_v2/weights/best.pt'
        if Path(v4_path).exists():
            model_path = v4_path
        elif Path(v3_path).exists():
            model_path = v3_path
        else:
            model_path = v2_path

    model = YOLO(model_path)
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "="*60)
    print("📷 WEBCAM CLASSIFICATION (Optimized FPS + Center Guide)")
    print("="*60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save frame")
    print("  'f' - Toggle FPS display")
    print("  'b' - Toggle background blur")
    print("="*60)
    print(f"🤖 Model: {Path(model_path).name}")
    print("="*60 + "\n")

    show_fps = True
    frame_count = 0
    import time
    start_time = time.time()

    # ⚡ Warm-up: Run a quick forward pass to initialize CUDA/graph
    print("⚙️ Warming up model...")
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=IMG_SIZE, conf=0.2, verbose=False)
    print("✅ Warm-up complete!\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        process_frame = frame

        # === Background blur (optional) ===
        if blur_background:
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            center_h, center_w = int(h * 0.7), int(w * 0.7)
            start_h, start_w = (h - center_h) // 2, (w - center_w) // 2
            cv2.ellipse(mask, (w//2, h//2), (center_w//2, center_h//2),
                        0, 0, 360, 255, -1)
            blurred = cv2.GaussianBlur(frame, (41, 41), 0)
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            process_frame = (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)

        # === Downscale slightly before prediction to speed up ===
        small_frame = cv2.resize(process_frame, (IMG_SIZE, IMG_SIZE))

        # === Faster prediction: disable plotting and use stream mode ===
        results = model.predict(
            source=small_frame,
            imgsz=IMG_SIZE,
            conf=0.25,
            verbose=False,
            stream=False,
            show=False,
            save=False
        )

        result = results[0]
        top1_class = result.names[result.probs.top1]
        top1_conf = result.probs.top1conf.item()

        # === Overlay information ===
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        center_h, center_w = int(h * 0.6), int(w * 0.6)
        start_h, start_w = (h - center_h) // 2, (w - center_w) // 2
        end_h, end_w = start_h + center_h, start_w + center_w

        # Draw guide box
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (start_w, start_h), (end_w, end_h), (0, 255, 255), 2)
        cv2.putText(overlay, "Place object here (center focus)",
                    (start_w + 20, start_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        display_frame = cv2.addWeighted(display_frame, 0.9, overlay, 0.1, 0)

        # Color coding
        if top1_conf > 0.7:
            color = (0, 255, 0)
            status = "HIGH"
        elif top1_conf > 0.5:
            color = (0, 255, 255)
            status = "MEDIUM"
        else:
            color = (0, 165, 255)
            status = "LOW"

        # Prediction text
        cv2.putText(display_frame, f"{top1_class.upper()}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(display_frame, f"{top1_conf:.1%} - {status}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # FPS calculation
        if show_fps:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(display_frame, f"FPS: {fps:.1f}",
                        (display_frame.shape[1] - 160, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if blur_background:
            cv2.putText(display_frame, "BG BLUR: ON", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Office Supplies Classifier (High FPS + Center Box)', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved: {filename}")
        elif key == ord('f'):
            show_fps = not show_fps
        elif key == ord('b'):
            blur_background = not blur_background
            print(f"🔄 Background blur: {'ON' if blur_background else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Webcam closed")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8 - Final Version')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'validate', 'predict', 'webcam'],
                       help='Mode to run')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image')
    parser.add_argument('--model', type=str, default=None,
                       help='Model path')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID')
    parser.add_argument('--preprocess', action='store_true',
                       help='Apply center-focus preprocessing')
    parser.add_argument('--blur-bg', action='store_true',
                       help='Blur background in webcam')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("\n🚀 FINAL TRAINING - Fixing background + pen/pencil issues")
        print("💡 Strategy:")
        print("   1. Heavy crops (50%) - Forces focus on CENTER object")
        print("   2. Heavy erasing (60%) - Learns from partial views")
        print("   3. Extreme colors - Ignores color cues")
        print("   4. High resolution (384) - Captures pen tip details")
        print("\n⏱️  Expected time: 6-7 hours")
        print("🎯 Should fix BOTH background AND pen/pencil issues!\n")
        model = train_model()
        print("\n✅ Training done! Test with problematic images")
        
    elif args.mode == 'validate':
        validate_model(args.model)
        
    elif args.mode == 'predict':
        if not args.image:
            print("❌ Error: --image required")
        else:
            predict_image(args.image, args.model, use_preprocessing=args.preprocess)
            
    elif args.mode == 'webcam':
        predict_webcam(args.model, args.camera, args.blur_bg)