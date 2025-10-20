# Import all dependencies.
import tkinter as tk
from tkinter import filedialog
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
RESULTFONT = ("Arial", 16) # Define the font parameters for the result label.
FRAME_SIZE = 224 # Size of image size frame



# Define class to handle an uploaded image using a machine learning model to perform object classification.
class UploadImage:
    def __init__(self, root : tk.Tk, model_service : any):
        self.root = root
        self.model_service = model_service
        
        # Create a separate top-level window for upload image.
        self.window = tk.Toplevel(root)
        self.window.title("Office Item Classifier - Upload Image Detection.")
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)
        
        # Configure size & center.
        WindowConfig.configure_window(self.window)
        
        # Label widget to display in the window frame.
        self.image_label = tk.Label(self.window)
        self.image_label.pack()
        
        # Label widget to show the result of the uploaded display.
        self.result_label = tk.Label(self.window, text="", font=RESULTFONT)
        self.result_label.pack(pady=5)

        # Upload button.
        self.upload_btn = tk.Button(self.window, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=5)
        
        # Button to go back to the main menu.
        self.back_btn = tk.Button(self.window, text="Back", command=self.close_window)
        self.back_btn.pack(pady=5)


    # Process an image for model prediction.
    def process_image(self, image_path ) -> np.ndarray:
        image = cv2.imread(image_path) # Read image using OpenCV.
        image = cv2.resize(image, (FRAME_SIZE, FRAME_SIZE)) # Resize to model input size.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR (OpenCV) to RGB.
        image = image.astype("float32") / 255.0 # Normalise pixel values.
        return np.expand_dims(image, axis=0), image # Expand dims for batch size and return original for display.

    
    # Allow a user to input an image and predict what image is.
    def upload_image(self) -> None:
        
        # Check if the image upload is the required format, else if no file is selected exit the function.
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        # Process image for model.
        image_input, image_display = self.process_image(file_path)
        
        # Run prediction using the model service.
        _, class_index, confidence = self.model_service.predict_image(image_input)
        
        # Add the label with prediction to the frame.
        label = f"{CLASS_NAMES[class_index]} ({confidence*100:.1f}%)"
        self.result_label.config(text=label)

        # Convert image back to a displayable format using PIL.
        image_pil = Image.fromarray((image_display*255).astype(np.uint8))
        image_tk = ImageTk.PhotoImage(image_pil)
        
        # Display image in Tkinter label.
        self.image_label.image_tk = image_tk
        self.image_label.config(image=image_tk)


    # Close the Tkinter window.
    def close_window(self) -> None:
        self.window.destroy()