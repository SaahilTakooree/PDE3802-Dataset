# Import all dependencies.
import tkinter as tk
from camera_view import CameraView
from gui_utils import WindowConfig
from upload_image import UploadImage
from stats_view import StatsView



# Class to handle the main menu GUI for office item Classifier application.
class MainMenu:
    
    # Initialse the main menu menu for button for camera detection, image upload, stats and quit.
    def __init__(self, root : tk.Tk, model_service : any):
        self.root = root
        self.model_service = model_service
        
        # Set the window title.
        self.root.title("Office Item Classifier - Main Menu")
        
        # Configure size & center.
        WindowConfig.configure_window(self.root)
    
        # Button for live camera detection.
        camera_button = tk.Button(root, text="Live Camera Detection", command=self.open_camera)
        camera_button.pack(pady=10)
        
        # Button for image detection.
        upload_button = tk.Button(root, text="Upload Image Detection", command=self.open_upload)
        upload_button.pack(pady=10)
        
        # Button to view model statistics.
        stats_button = tk.Button(root, text="View Model Stats", command=self.view_stats)
        stats_button.pack(pady=10)
        
        # Button to quit application.
        close_button = tk.Button(root, text="Quit", command=root.quit)
        close_button.pack(pady=10)
    
    
    # Open camera view window for live detection.
    def open_camera(self):
        CameraView(self.root, self.model_service)
    
    
    # Upload image for upload detection.
    def open_upload(self):
        UploadImage(self.root, self.model_service)
    
    # Display the stats.
    def view_stats(self):
        StatsView(self.root)



# Initialise and run the main application window.
if __name__ == "__main__":
    from model_service import ModelService # Import the model service for interface.
    root = tk.Tk() # Create the root Tkinter window.
    model_service = ModelService() # Initialise the model service.
    app = MainMenu(root, model_service) # Create a main menu.
    root.mainloop() # Start the tkinter event loop.