# Import dependencies.
import torch # Import 'PyTorch' for a deep learning framework to manage devices and model prediction.
from ultralytics import YOLO # Import 'Ultralytics' to load object classification model.
import numpy as np # Import 'Numpy' to create and manipulate image arrays.
from typing import Tuple, Optional, Union  # For type hints.



# Model service to manage loading, preparing, and running of YOLO model.
class ModelService:

    # Initialise the 'ModelService'.
    def __init__(self, model_path : str) -> None:

        # Display a message to show in the terminal to that model has started to be loaded into the memory.
        print("Loading model... Please wait.")

        # Detecting which device is avaliable.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load the Yolo Model into memory.
        self.model = YOLO(model_path)

        # Move model to the seleted devices.
        self.model.to(self.device)

        # If GPU is avaliable, use half-precision to improve speed and reduce memory usage.
        if self.device == 'cuda':
            self.model.half()

        # Warm up the model using a dummy webcam-sized frame (To prevent first use lag).
        self._warm_up_model()
        
        # Display a message to show in the terminal to that model has been loaded into the memory.
        print("Model loaded. Starting GUI.")


    # Function to warm up the model to prevent first time lag during first time prediction during first loading.
    def _warm_up_model(self) -> None:
        
        # Create a blank dummy image.
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Perform a prediction on a dummy frame for a still-image mode.
        _ = self.model.predict(dummy_frame, imgsz=224, conf=0.25, verbose=False, device=self.device)

        # Perform a prediction on a dummy frame for a webcam input.
        _ = self.model.predict(source=dummy_frame, stream=False, verbose=False, device=self.device)


   # Function to run a YOLO prediction on a give frame.
    def predict(self, image: Union[np.ndarray, str], imgsz: Optional[int]=None) -> Tuple[str, float]:
        
        # Get the image size dynamically based on input type.
        if imgsz is None:
            imgsz = 224 if isinstance(image, np.ndarray) else 384
        
        # Perform the YOLO detection using the model.
        results = self.model.predict(image, imgsz=imgsz, conf=0.6, verbose=False, device=self.device)
        
        # Get the first result.
        result = results[0]
        
        # Get the class label from the result.
        pred_class = result.names[result.probs.top1]
        
        # Get the confiance from the result.
        pred_conf = float(result.probs.top1conf.item())
        
        # Return the class and confiance.
        return pred_class, pred_conf