# Import all dependencies.
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from gui_utils import WindowConfig



# Defining of constant.
CLASS_NAMES = [
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
]  # List of class names that the model can predict
FRAME_SIZE = 224 # Size of camera size frame
UPDATE_INTERVAL_MS = 30 # The update interval of the camera frame.
LABEL_POSITION = (10, 30) # The position where to put the label in the frame.
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX # The label font.
LABEL_SCALE = 1 # The font scale.
LABEL_COLOR = (0, 255, 0) # The colour of the label.
LABEL_THICKNESS = 2 # The thickness of the label.


# Define class to handle a live camera feed using a machine learning model to perform object classification on each frame in real-time.
class CameraView:
    
    # Initialises the CameraView window and start video capture.
    def __init__(self, root : tk.Tk, model_service : any, camera_index : int = 0):
        self.root = root
        self.model_service = model_service
        
        # Create a separate top-level window for live camera.
        self.window = tk.Toplevel(root)
        self.window.title("Office Item Classifier - Live Camera Detection.")
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)
        
        # Configure size & center.
        WindowConfig.configure_window(self.window)
        
        # Label widget to display in the window frame.
        self.video_label = tk.Label(self.window)
        self.video_label.pack()
        
        # Button to go back to the main window.
        self.back_button = tk.Button(self.window, text="Back", command=self.close_window)
        self.back_button.pack(pady=5)
        
        # Initialise the video capture with default camera.
        self.capture = cv2.VideoCapture(camera_index)
        # If camera is not open, close the window.
        if not self.capture.isOpened():
            messagebox.showerror("Error", f"Cannot open camera {self.camera_index}.")
            self.window.destroy()
            return
        
        # Start the update frame loop.
        self.update_frame()
        
    
    # Process a frame for model prediction.
    def process_frame(self, frame : np.ndarray) -> np.ndarray:
        image = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE)) # Resize the frame to 224x224.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR (OpenCV default) to RGB.
        image = image.astype("float32") / 255.0 # Normalise pixel values to [0, 1].
        return np.expand_dims(image, axis=0) # Expand dimensions to create a batch of size 1.
    
    
    # Capture a frame from live video feed and predict the object class.
    def update_frame(self) -> None:
        
        # Get the current frame and if the capturing the frame was successful.
        return_value, frame = self.capture.read()
        
        if return_value:
            
            # Process the frame for the model prediction.
            image_input = self.process_frame(frame)
            
            # Predict the class index and confidence
            _, class_index, confidence = self.model_service.predict_image(image_input)
            
            # Create a label for the class name and confidence.
            label = f"{CLASS_NAMES[class_index]} ({confidence*100:.1f}%)"
            
            # Draw the label on the frame.
            cv2.putText(frame, label, LABEL_POSITION, LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICKNESS, cv2.LINE_AA)
            
            # Convert BGR frame to RGB for Tkinter display.
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_tk = ImageTk.PhotoImage(image=image_pil)
            
            # Update the Tkinter label with the new frame.
            self.video_label.image_tk = image_tk
            self.video_label.configure(image=image_tk)
        
        # Schedule the next frame update.
        self.window.after(UPDATE_INTERVAL_MS, self.update_frame)
    
    
    # Release the camera resources and close the Tkinter window.
    def close_window(self) -> None:
        self.capture.release()
        self.window.destroy()