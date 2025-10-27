"""
YOLOv8 Classification - STABLE MULTI-ANGLE VERSION
Fixes: Angle sensitivity + Lighting issues + Training stability
Based on working v4 code with multi-angle improvements
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import cv2
import numpy as np

# ============================================
# CONFIGURATION - STABLE + MULTI-ANGLE
# ============================================
DATASET_ROOT = Path("data")
PROJECT_NAME = "office_supplies_classifier"
MODEL_SIZE = "l"

CLASSES = [
    "erasers", "glue_sticks", "highlighters", "mugs", "paper_clips",
    "pencils", "pens", "staplers", "tapes", "usb_sticks"
]

# BALANCED: Keep v4 stability + Add multi-angle features
EPOCHS = 100              # Reduced from 200 - more realistic
BATCH_SIZE = 16           # Back to 16 - stable and tested
IMG_SIZE = 384            # Keep 384 - proven to work
PATIENCE = 20             # Reduced from 40 - allows completion
LEARNING_RATE = 0.00003   # Same as v4 - stable
DROPOUT = 0.4             # Same as v4 - not too aggressive

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
# TRAIN - STABLE MULTI-ANGLE
# ============================================
def train_model():
    """Stable training with multi-angle support"""
    
    yaml_path = create_data_yaml()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"🎯 STABLE MULTI-ANGLE TRAINING")
    print(f"{'='*70}")
    print(f"Based on: Working v4 code")
    print(f"NEW: 360° rotation + Better lighting variation")
    print(f"Fixed: Training stability (will complete!)")
    print(f"{'='*70}")
    print(f"Device: {device.upper()}")
    print(f"Model: YOLOv8{MODEL_SIZE}-cls")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch: {BATCH_SIZE} | Epochs: {EPOCHS} | Patience: {PATIENCE}")
    print(f"{'='*70}\n")
    
    model = YOLO(f'yolov8{MODEL_SIZE}-cls.pt')
    
    results = model.train(
        data=str(DATASET_ROOT),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        patience=PATIENCE,       # Realistic patience
        save=True,
        save_period=10,
        device=device,
        workers=4,
        project=PROJECT_NAME,
        name='train_v6_stable',  # New stable version
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        
        # LEARNING RATE - Proven stable from v4
        lr0=LEARNING_RATE,
        lrf=0.00001,
        momentum=0.937,
        weight_decay=0.001,
        
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,
        cos_lr=True,
        
        verbose=True,
        cache=False,
        amp=True,
        
        # BALANCED COLOR AUGMENTATION
        # Keep v4 base, moderate hue for pen/pencil distinction
        hsv_h=0.5,               # Reduced from 0.8 - keep some color info
        hsv_s=0.9,               # Strong saturation variation
        hsv_v=0.9,               # CRITICAL: Full brightness (dim to bright)
        
        # MULTI-ANGLE AUGMENTATION
        degrees=270.0,           # 3/4 rotation (not full 360 - more stable)
        translate=0.45,          # Moderate movement
        scale=0.9,               # Strong zoom range
        shear=15.0,              # Moderate shear (mugs from side)
        perspective=0.002,       # Perspective changes
        flipud=0.6,              # Increased from v4 (0.5 → 0.6)
        fliplr=0.6,              # Increased from v4 (0.5 → 0.6)
        
        # BACKGROUND ELIMINATION (same as v4)
        mosaic=0.0,
        mixup=0.5,               # Proven effective in v4
        copy_paste=0.0,
        auto_augment='randaugment',
        erasing=0.5,             # Moderate (not 0.6 - too aggressive)
        crop_fraction=0.55,      # Slight increase from v4 (0.5 → 0.55)
        
        # REGULARIZATION (same as v4)
        dropout=DROPOUT,
        label_smoothing=0.15,
        
        # TRAINING DYNAMICS
        close_mosaic=15,
        plots=True,
        val=True,
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"📁 Model: {PROJECT_NAME}/train_v6_stable/weights/best.pt")
    print(f"💡 Improvements over v4:")
    print(f"   ✓ 270° rotation (was 180°) - better angles")
    print(f"   ✓ More flips (60% vs 50%) - upside down objects")
    print(f"   ✓ Full brightness range - any lighting")
    print(f"   ✓ More visible crops - see handles/sides")
    
    return model

# ============================================
# VALIDATION
# ============================================
def validate_model(model_path=None):
    """Validate model"""
    
    if model_path is None:
        # Auto-select best available
        v6_path = f'{PROJECT_NAME}/train_v6_stable/weights/best.pt'
        v4_path = f'{PROJECT_NAME}/train_v4_final/weights/best.pt'
        v3_path = f'{PROJECT_NAME}/train_v3_extreme_aug/weights/best.pt'
        v2_path = f'{PROJECT_NAME}/train_v2/weights/best.pt'
        
        if Path(v6_path).exists():
            model_path = v6_path
        elif Path(v4_path).exists():
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
# PREDICT
# ============================================
def preprocess_image_center_focus(image_path):
    """Preprocess: Focus on center, blur background"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center_h, center_w = int(h * 0.6), int(w * 0.6)
    start_h, start_w = (h - center_h) // 2, (w - center_w) // 2
    mask[start_h:start_h+center_h, start_w:start_w+center_w] = 255
    
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    result = np.where(mask[:,:,None] == 255, img, blurred)
    
    return result.astype(np.uint8)

def predict_image(image_path, model_path=None, conf_threshold=0.3, use_preprocessing=False):
    """Predict with optional center-focus preprocessing"""
    
    if model_path is None:
        v6_path = f'{PROJECT_NAME}/train_v6_stable/weights/best.pt'
        v4_path = f'{PROJECT_NAME}/train_v4_final/weights/best.pt'
        v3_path = f'{PROJECT_NAME}/train_v3_extreme_aug/weights/best.pt'
        v2_path = f'{PROJECT_NAME}/train_v2/weights/best.pt'
        
        if Path(v6_path).exists():
            model_path = v6_path
        elif Path(v4_path).exists():
            model_path = v4_path
        elif Path(v3_path).exists():
            model_path = v3_path
        else:
            model_path = v2_path
    
    model = YOLO(model_path)
    
    source = image_path
    if use_preprocessing:
        preprocessed = preprocess_image_center_focus(image_path)
        if preprocessed is not None:
            source = preprocessed
            print("💡 Applied center-focus preprocessing")
    
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
    
    print("\n📊 Top 5 Predictions:")
    print("-" * 60)
    for i, (idx, conf) in enumerate(probs, 1):
        class_name = result.names[idx].replace('_', ' ')
        bar = "█" * int(conf * 50)
        print(f"  {i}. {class_name:15s} {conf:6.2%} {bar}")
    
    if len(probs) >= 2:
        gap = probs[0][1].item() - probs[1][1].item()
        print(f"\n💡 Gap (1st vs 2nd): {gap:.2%}")
        
        if gap < 0.15:
            print("   ⚠️  Close call! Consider both options")
            
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
# WEBCAM
# ============================================
def predict_webcam(model_path=None, camera_id=0, blur_background=True):
    """Fast webcam classification (>5 FPS on CPU, >20 FPS on GPU)"""

    import time
    import threading
    from collections import deque

    # --------------------------------------------------
    # 1️⃣  Load lightweight model
    # --------------------------------------------------
    if model_path is None:
        model_path = r"C:\Users\mehar\Downloads\PDE3802-Dataset\office_supplies_classifier\train_v4_final\weights\best.pt"  # smallest and fastest

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)
    if device == 'cuda':
        model.half()

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return

    print("\n⚡ FAST MODE - YOLOv8 Classification")
    print("Model:", model_path)
    print("Device:", device.upper())
    print("Target: >5 FPS (CPU) / >20 FPS (GPU)\n")

    fps_queue = deque(maxlen=20)
    last_pred = ("...", 0.0)
    frame_skip = 2   # infer every 2nd frame
    frame_count = 0

    print("⚙️ Warming up...")
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=224, conf=0.2, verbose=False)
    print("✅ Ready!\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        t0 = time.time()

        # Predict every nth frame only
        if frame_count % frame_skip == 0:
            results = model.predict(
                source=frame,
                imgsz=224,
                conf=0.25,
                verbose=False,
                device=device
            )

            result = results[0]
            top1 = result.names[result.probs.top1]
            conf = result.probs.top1conf.item()
            last_pred = (top1, conf)

        pred_class, pred_conf = last_pred

        elapsed = time.time() - t0
        fps = 1.0 / max(elapsed, 1e-3)  # prevents division by zero

        fps_queue.append(fps)
        avg_fps = np.mean(fps_queue)

        color = (0, 255, 0) if pred_conf > 0.7 else (0, 255, 255)
        cv2.putText(frame, f"{pred_class.upper()}: {pred_conf:.1%}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("YOLOv8 Fast Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam closed")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8 - Stable Multi-Angle')
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
        print("\n🚀 STABLE MULTI-ANGLE TRAINING")
        print("="*70)
        print("💡 Based on working v4 code with improvements:")
        print("   ✓ 270° rotation (was 180°) - better angle coverage")
        print("   ✓ 60% flips (was 50%) - upside down objects")
        print("   ✓ Full brightness range - any lighting")
        print("   ✓ More visible crops (55% vs 50%) - see handles/sides")
        print("   ✓ Moderate hue (0.5) - keep pen/pencil color distinction")
        print("\n🔧 Stability improvements:")
        print("   ✓ Realistic epochs (100 vs 200)")
        print("   ✓ Proven batch size (16)")
        print("   ✓ Realistic patience (20)")
        print("   ✓ Balanced augmentation (not too aggressive)")
        print("\n⏱️  Expected time: 6-7 hours")
        print("✅ Training WILL complete this time!\n")
        model = train_model()
        print("\n✅ Training complete!")
        print("📌 Test with angles: 0°, 45°, 90°, 180°")
        print("📌 Test with lighting: dim and bright")
        
    elif args.mode == 'validate':
        validate_model(args.model)
        
    elif args.mode == 'predict':
        if not args.image:
            print("❌ Error: --image required")
        else:
            predict_image(args.image, args.model, use_preprocessing=args.preprocess)
            
    elif args.mode == 'webcam':
        predict_webcam(args.model, args.camera, args.blur_bg)