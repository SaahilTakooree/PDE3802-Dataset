# Import dependencies.
import customtkinter as ctk # Import 'CustomTkinter' for modern looking tkinter GUI.
from gui.services.model_service import ModelService # Import 'ModelService' class for type hinting.
import cv2 # Import 'OpenCV' for real time computer vision.
import threading # 'Threading' to allow camera operatin to run in the background.
from PIL import Image # Import 'Image' to handle image loading and resizing.
from customtkinter import CTkImage # Import 'CTkImage' to display image in CTk widget.



# Class to handle the live object classfication using a webcam.
class CameraTab(ctk.CTkFrame):
    
    # Initialise the camera tab.
    def __init__(self, parent : ctk.CTkBaseClass, model_service : ModelService) -> None:

        # Initialise the CTk Frame.
        super().__init__(parent, fg_color = "transparent")
        
        # Make sure the frame fill all the space available.
        self.pack(fill = "both", expand = True)
        
        # Store a reference of the model service.
        self.model_service = model_service

        # Create a main frame container.
        self.main_frame = ctk.CTkFrame(self, fg_color = "transparent")
        self.main_frame.pack(fill = "both", expand = True, padx = 20, pady = 20) # Add padding to main frame container and make sure it take up all of the avaliable space.

        # Create a camera display frame.
        self.cam_frame = ctk.CTkFrame(
            self.main_frame,
            corner_radius = 15,
            border_width = 2,
            border_color = "#2b2b2b",
            fg_color = "#1a1a1a"
        )
        self.cam_frame.pack(side = "left", padx = 10, pady = 10, expand = True, fill = "both") # Add padding to camera frame and make sure it take up all of the avaliable space.

        # Label to show the camera feed or placeholder text.
        self.cam_label = ctk.CTkLabel(
            self.cam_frame,
            text = "Camera feed will appear here\n\nClick 'Start Camera' to begin",
            font = ("Segoe UI", 16),
            text_color = "#606060"
        )
        self.cam_label.pack(expand = True) # Allow the camera label to expand as needed.

        # Create a camera control frame
        self.btn_frame = ctk.CTkFrame(
            self.main_frame,
            corner_radius = 15,
            fg_color = "#1a1a1a",
            border_width = 2,
            border_color = "#2b2b2b"
        )
        self.btn_frame.pack(side = "right", padx = 10, pady = 10, anchor = "n", fill = "y") # Add padding to the camera controll frame.

        # Title label for the control section.
        self.control_title = ctk.CTkLabel(
            self.btn_frame,
            text = "Controls",
            font = ("Segoe UI", 18, "bold")
        )
        self.control_title.pack(pady = (20, 15)) # Add padding to the controll section.

        # Button to start the camera feed.
        self.start_cam_btn = ctk.CTkButton(
            self.btn_frame,
            text = "Start Camera",
            command = self.start_camera,
            font = ("Segoe UI", 14, "bold"),
            height = 45,
            width = 180,
            corner_radius = 10,
            fg_color = "#16a34a",
            hover_color = "#15803d"
        )
        self.start_cam_btn.pack(pady = (0, 15), padx = 20) # Add padding to the start camera feed button.

        # Button to stop the camera feed.
        self.stop_cam_btn = ctk.CTkButton(
            self.btn_frame,
            text = "Stop Camera",
            command=self.stop_camera,
            font = ("Segoe UI", 14, "bold"),
            height = 45,
            width = 180,
            corner_radius = 10,
            fg_color = "#dc2626",
            hover_color = "#b91c1c"
        )

        # Create a status frame.
        self.status_frame = ctk.CTkFrame(
            self.btn_frame,
            corner_radius = 10,
            fg_color = "#0a0a0a"
        )
        self.status_frame.pack(pady = (20, 15), padx = 20, fill = "x") # Add padding to the status frame.

        # Label to show current camera status.
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text = "Status: Inactive",
            font = ("Segoe UI", 12, "bold"),
            text_color = "#606060"
        )
        self.status_label.pack(pady = 10) # Add padding to the current camera status label.

        # Create an info frame.
        self.info_frame = ctk.CTkFrame(
            self.btn_frame,
            corner_radius = 10,
            fg_color = "#0a0a0a"
        )
        self.info_frame.pack(pady = (10, 20), padx = 20, fill = "both", expand = True) # Add padding to the info frame.

        # Label showing information about confidence levels.
        self.info_label = ctk.CTkLabel(
            self.info_frame,
            text = "Real-time Detection\n\nConfidence:\n > 70% - High\n < 70% - Low",
            font = ("Segoe UI", 11),
            justify = "left",
            text_color = "#808080"
        )
        self.info_label.pack(pady = 15, padx = 10) # Add padding to the label information above confidence levels.
        
        # Check if camera is available.
        if not self._check_camera_available():
            # Update label to show no camera.
            self.cam_label.configure(text="No camera detected.\nPlease connect a webcam and restart application.")
            # Disable start camera button.
            self.start_cam_btn.configure(state="disabled", fg_color="#555555")

        # Camera variables.
        self.capture = None # Store the OpenCV video capture object.
        self.running = False # Flag to indicate if the camera is running.
        self.thread = None # Thread for updating the camera feed.
        self.model = self.model_service.model # Model used for prediction.
        self.device = self.model_service.device # Device used for inference.


    # Function to update the UI and status text based on the camera state.
    def _update_ui_state(self, running: bool) -> None:
        # When camera is active.
        if running:
            self.start_cam_btn.pack_forget()
            self.stop_cam_btn.pack(pady = (0, 15), padx = 20)
            self.cam_label.configure(text= "")
            self.status_label.configure(
                text = "Status: Active",
                text_color = "#16a34a"
            )
            self.update_idletasks()
        else: # When camera is stop.
            self.stop_cam_btn.pack_forget()
            self.start_cam_btn.configure(text = "Restart Camera")
            self.start_cam_btn.pack(pady = (0, 15), padx = 20)
            self.status_label.configure(
                text="Status: Inactive",
                text_color = "#606060"
            )
            self.update_idletasks()


    # Function to start the camera feed in a background thread.
    def start_camera(self) -> None:
        
        # If the camera is running no need to restart it.
        if self.running:
            return

        # Set the flag to true to know the the camera is running.
        self.running = True
        
        # Update the UI to active state.
        self._update_ui_state(self.running)

        # Initialise the camera using OpenCV
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # If camera fails to open revert states.
        if not self.capture.isOpened():
            self.running = False
            self._update_ui_state(self.running)
            return

        # Start thread to continuously update the camera frames.
        self.thread = threading.Thread(target=self._update_camera, daemon = True)
        self.thread.start()


    # Function to stop the camera feed and release resources.
    def stop_camera(self) -> None:
        
        # If the camera is already stop, exit.
        if not self.running:
            return

        # Update the flag to indicate that the camera has stop.
        self.running = False
        
        # Release the camera capture if it is active.
        if self.capture:
            self.capture.release()
            self.capture = None

        # Update the UI to inactive state.
        self._update_ui_state(self.running)


    # Function to Continuously read franes from camera and update display.
    def _update_camera(self) -> None:
        
        # Skip every 2nd frame for performance.
        frame_skip = 2
        
        # Keep track of processed frames.
        frame_count = 0
        
        # Variable to store the last prediction.
        last_pred = ("None", 0.0)

        # Loop while the camera is active.
        while self.running and self.capture.isOpened():
            
            # Store the current frame.
            ret, frame = self.capture.read()
            if not ret:
                break

            # Flip frame to have a mirror-like view.
            frame = cv2.flip(frame, 1)
            
            # Increase frame count.
            frame_count += 1

            # Perform prediction every few frames.
            if frame_count % frame_skip == 0:
                last_pred = self.model_service.predict(frame)

            # Update prediction.
            pred_class, pred_conf = last_pred
            
            # Choose text colour based on confidence.
            color = (0, 255, 0) if pred_conf > 0.7 else (0, 255, 255)

            # Draw the class name and confidence on the frame.
            cv2.putText(frame, f"{pred_class.upper()}: {pred_conf:.1%}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # Convert frame to RGB and display it.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            ctk_image = CTkImage(light_image=img_pil, dark_image=img_pil, size=(640, 480))
            self.cam_label.configure(image=ctk_image)
            self.cam_label.image = ctk_image

        # Release camera when loops ends.
        if self.capture:
            self.capture.release()


    # Function to check if there is a camera.
    def _check_camera_available(self) -> bool:
        test_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if test_cap.isOpened():
            test_cap.release()
            return True
        return False