# gui/upload_view.py
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

FRAME_SIZE = 224
CLASS_NAMES = [
    "erasers", "glue_sticks", "highlighters", "mugs", "paper_clips",
    "pencils", "pens", "staplers", "tapes", "usb_sticks"
]

class UploadViewFrame(ctk.CTkFrame):
    def __init__(self, parent, model_service, on_back):
        super().__init__(parent)
        self.model_service = model_service
        self.on_back = on_back

        top = ctk.CTkFrame(self)
        top.pack(pady=12, padx=12, fill="x")

        ctk.CTkButton(top, text="Upload Image", command=self.upload_image).pack(side="left", padx=(0,8))
        ctk.CTkButton(top, text="Back", command=self.on_back).pack(side="left")

        self.image_label = ctk.CTkLabel(self, text="")
        self.image_label.pack(pady=12)

        self.result_label = ctk.CTkLabel(self, text="", wraplength=600, font=ctk.CTkFont(size=14))
        self.result_label.pack(pady=8)

    def process_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Unable to read image")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE))
        return resized, rgb

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        try:
            model_input, display_img = self.process_image(file_path)
            _, class_index, confidence = self.model_service.predict_image(model_input)
            label = f"{CLASS_NAMES[class_index]} ({confidence*100:.1f}%)"
            self.result_label.configure(text=label)
            # show preview
            pil = Image.fromarray(display_img)
            pil = pil.resize((480, 360))
            tkimg = ImageTk.PhotoImage(pil)
            self.image_label.configure(image=tkimg)
            self.image_label.image = tkimg
        except Exception as e:
            self.result_label.configure(text=f"Error: {e}")
