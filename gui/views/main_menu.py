# Import dependencies.
import customtkinter as ctk # Import 'CustomTkinter' for modern looking tkinter GUI.
from gui.services.model_service import ModelService # Import 'ModelService' class for type hinting.
from gui.views.upload_tab import UploadTab # Import the upload tab view for image classification.
from gui.views.camera_tab import CameraTab # Import the camera tab for lice camera view.
from gui.views.stats_tab import StatsTab # Import for the stats tab to display the prediction statistics.



# Class the represent the primary GUI for the application.
class MainMenu(ctk.CTkFrame):

    # Initialise the MainMenu GUI frame.
    def __init__(self, parent : ctk.CTk, model_service : ModelService) -> None:

        # Transparent background for the main frame.
        super().__init__(parent, fg_color = "transparent")

        # Store the model service.
        self.model_service = model_service

        # Make the frame fell the parent window.
        self.pack(fill = "both", expand = True)

        # Create a header for with a label.
        self.header_frame = ctk.CTkFrame(self, height = 50, corner_radius = 0, fg_color = "#1a1a1a") # Define the header.
        self.header_frame.pack(fill = "x", pady = (0, 2)) # Set padding.
        self.header_frame.pack_propagate(False) # Prevent resizing in children frame.

        self.label = ctk.CTkLabel(self.header_frame, text = "Office Supplies Classifier",  font = ("Segoe UI", 24, "bold"), text_color = "#ffffff") # Set the header label.
        self.label.pack(pady = 5) # Set the label padding.

        # Create tab view.
        self.tabs = ctk.CTkTabview(
            self,
            corner_radius = 15,
            border_width = 2,
            segmented_button_fg_color = "#2b2b2b",
            segmented_button_selected_color = "#1f6aa5",
            segmented_button_selected_hover_color = "#1a5a8e"
        ) # Define the tab.
        self.tabs.pack(fill="both", expand = True, padx = 25, pady = (0, 2)) # Set the padding for the tab.

        # Add the tabs.
        upload_tab = self.tabs.add("Upload Image")
        camera_tab = self.tabs.add("Camera Prediction")
        stats_tab = self.tabs.add("View Stats")
        
        # Set equal tab width button.
        for button in self.tabs._segmented_button._buttons_dict.values():
            button.configure(width=250)

        # Initialise tab views with the model service.
        self.upload_view = UploadTab(upload_tab, self.model_service)
        self.camera_view = CameraTab(camera_tab, self.model_service)
        self.stats_view = StatsTab(stats_tab)