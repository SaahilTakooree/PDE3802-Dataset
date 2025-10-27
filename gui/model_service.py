# gui/model_service.py
from ultralytics import YOLO
import numpy as np

FRAME_SIZE = 224

class ModelService:
    """
    Wrapper for YOLOv8 classification model.
    Exposes predict_image(image: np.ndarray) -> (probs, class_index, confidence)
    image: numpy array HxWx3 in RGB (uint8 or float)
    """
    def __init__(self, model_path: str = r"C:\Users\mehar\Downloads\PDE3802-Dataset\office_supplies_classifier\train_v4_final\weights\best.pt"):
        print(f"[ModelService] Loading YOLO model from {model_path} ...")
        self.model = YOLO(model_path)
        print("[ModelService] Model loaded")

    def _prepare(self, image: np.ndarray):
        # convert to float32 and scale 0..1 if needed
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        return image


    def predict_image(self, image: np.ndarray):
        """
        Predict on a single image (HxWx3 numpy array, RGB).
        Returns (probs, class_index, confidence)
        """
        img = self._prepare(image)
        # run prediction - use imgsz to match training input
        results = self.model.predict(source=img, imgsz=FRAME_SIZE, verbose=False)

        # results is a list-like of Results objects; take first
        r = results[0]

        # For classification models, ultralytics usually provides r.probs (np.ndarray)
        try:
            probs = r.probs.numpy()  # <- convert to ndarray
        except Exception:
            probs = np.ones((1,), dtype=float)
        class_index = int(np.argmax(probs))
        confidence = float(np.max(probs))
        return probs, class_index, confidence
