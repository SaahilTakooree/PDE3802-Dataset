# gui/camera_view.py
import customtkinter as ctk
from customtkinter import CTkImage
import cv2
from PIL import Image, ImageTk
import numpy as np
from threading import Lock

FRAME_SIZE = 224
UPDATE_INTERVAL_MS = 30

CLASS_NAMES = [
    "erasers", "glue_sticks", "highlighters", "mugs", "paper_clips",
    "pencils", "pens", "staplers", "tapes", "usb_sticks"
]

class CameraViewFrame(ctk.CTkFrame):
    def __init__(self, parent, model_service, on_back):
        super().__init__(parent)
        self.model_service = model_service
        self.on_back = on_back

        # layout: left video, right info panel
        self.video_panel = ctk.CTkLabel(self, text="")
        self.video_panel.pack(side="left", padx=12, pady=12, expand=True, fill="both")

        right_panel = ctk.CTkFrame(self, width=300)
        right_panel.pack(side="right", padx=12, pady=12, fill="y")

        self.result_label = ctk.CTkLabel(right_panel, text="No prediction yet", wraplength=260)
        self.result_label.pack(pady=8)

        ctk.CTkButton(right_panel, text="Back", command=self.on_back).pack(pady=(10,5))
        ctk.CTkButton(right_panel, text="Stop Camera", command=self.stop_capture).pack(pady=5)
        self._cap = None
        self._running = False
        self._lock = Lock()

    def start_capture(self, camera_index=0):
        with self._lock:
            if self._running:
                return
            self._cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
            if not self._cap.isOpened():
                self.result_label.configure(text="Unable to open camera.")
                return
            self._running = True
            self._loop()

    def stop_capture(self):
        with self._lock:
            self._running = False
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

    def _loop(self):
        if not self._running or self._cap is None:
            return
        ret, frame = self._cap.read()
        if ret:
            # prepare frame for model (RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_for_model = cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE))
            # predict
            try:
                _, class_index, confidence = self.model_service.predict_image(img_for_model)
                label = f"{CLASS_NAMES[class_index]} ({confidence*100:.1f}%)"
            except Exception as e:
                label = f"Prediction error: {e}"

            # draw label on frame (display)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # convert and show
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_pil = image_pil.resize((640, 480))
            image_ctk = CTkImage(light_image=image_pil, dark_image=image_pil, size=(640,480))
            self.video_panel.configure(image=image_ctk)
            self.video_panel.image = image_ctk
            self.result_label.configure(text=label)

        # schedule next
        self.after(UPDATE_INTERVAL_MS, self._loop)
