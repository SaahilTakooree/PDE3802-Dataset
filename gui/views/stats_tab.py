# Import dependencies.
import customtkinter as ctk # Import 'CustomTkinter' for modern looking tkinter GUI.
from customtkinter import CTkLabel, CTkImage # Import 'CTkImage' and 'CTkLabel' to display image in CTk widget.
import os # Import 'os' to get system specific function.
from PIL import Image # Import 'Image' to handle image loading and resizing.



# Class to handle the displaying of the model stats.
class StatsTab(ctk.CTkFrame):

    # Initialise the tab layout.
    def __init__(self, parent : ctk.CTkBaseClass) -> None:
        
        # Initialise the parent CTk frame with a transparent background.
        super().__init__(parent, fg_color = "transparent")
        
        # Make the frame fill out all avaliable space.
        self.pack(fill = "both", expand = True, padx = 20)

        # Create a header label to show the title.
        self.header = CTkLabel(
            self,
            text = "Model Performance Metrics",
            font = ("Segoe UI", 18, "bold")
        )
        self.header.pack(pady = (0, 2)) # Add padding to the header.

        # Createa a tab widget.
        self.tab_view = ctk.CTkTabview(
            self,
            corner_radius = 15,
            border_width = 2,
            segmented_button_fg_color = "#2b2b2b",
            segmented_button_selected_color = "#1f6aa5",
            segmented_button_selected_hover_color = "#1a5a8e"
        )
        self.tab_view.pack(expand = True, fill = "both") # Allow the tab widget to fill out all avaliable space.
        
        # Add the tabs.
        self.tab_view.add("Confusion Matrix")
        self.tab_view.add("Results")
        self.tab_view.add("Metrics")
        
        # Set equal tab width button.
        for button in self.tab_view._segmented_button._buttons_dict.values():
            button.configure(width = 150)

        # Get the model data location.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

        # Define file path for model data.
        self.conf_matrix_path = os.path.join(project_root, "office_supplies_classifier", "train_v4_final", "confusion_matrix_normalized.png")
        self.results_path = os.path.join(project_root, "office_supplies_classifier", "train_v4_final", "results.png")
        self.metrics_path = os.path.join(project_root, "office_supplies_classifier", "train_v4_final", "metrics.txt")

        # Initialise all tab.
        self._init_image_tab(self.tab_view.tab("Confusion Matrix"), self.conf_matrix_path, "Confusion Matrix")
        self._init_image_tab(self.tab_view.tab("Results"), self.results_path, "Training Results")
        self._init_text_tab(self.tab_view.tab("Metrics"), self.metrics_path)


    # Function to initalise a tab that displays an image.
    def _init_image_tab(self, tab : ctk.CTkBaseClass, img_path : str, title : str) -> None:

        # Container frame for the image section.
        container = ctk.CTkFrame(tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10) # Add padding to image frame.

        # Title label for the image.
        title_label = CTkLabel(
            container,
            text = title,
            font = ("Segoe UI", 16, "bold")
        )
        title_label.pack(pady = (5, 5)) # Add vertical spacing to image label.

        # Add frame to hold the image.
        frame = ctk.CTkFrame(
            container,
            corner_radius = 15,
            border_width = 2,
            border_color = "#2b2b2b",
            fg_color ="#1a1a1a"
        )
        frame.pack(pady = 10, padx = 10, expand = True, fill = "both") # Add padding to image.

        # If image exists, display it.
        if os.path.exists(img_path):
            image = Image.open(img_path) # Load the image.
            label = CTkLabel(frame) # Label to show the image in.
            label.pack(expand = True, fill = "both", padx = 2, pady = 2) # Add padding to the image label.
            
            # Resize image for better display.
            resized = image.resize((450, 450), Image.LANCZOS)
            
            # Convert the image to CTkImage and display it.
            ctk_image = CTkImage(light_image = resized, dark_image = resized, size = (400, 400))
            label.configure(image = ctk_image)
            label.image = ctk_image
        else: # Else show and error message if the image is missing.
            error_label = CTkLabel(
                frame,
                text = f"Image not found:\n{img_path}",
                wraplength = 450,
                font = ("Segoe UI", 12),
                text_color ="#f87171"
            )
            error_label.pack(pady = 40)


    # Function to initialise a tab to display text.
    def _init_text_tab(self, tab : ctk.CTkBaseClass, txt_path : str):
        
        # Main container frame for the text.
        container = ctk.CTkFrame(tab, fg_color = "transparent")
        container.pack(fill = "both", expand = True, padx = 10, pady = 10) # Add padding to the container.

        # Title label above the text.
        title_label = CTkLabel(
            container,
            text = "Model Scores",
            font = ("Segoe UI", 16, "bold")
        )
        title_label.pack(pady = (0, 2)) # Add padding to the title label.

        # Frame that containe the text area.
        frame = ctk.CTkFrame(
            container,
            corner_radius = 15,
            border_width = 2,
            border_color ="#2b2b2b",
            fg_color = "#1a1a1a"
        )
        frame.pack(pady = 10, padx = 10, expand = True, fill = "both") 
        
        # Create a textbox widget to display the metrics.
        textbox = ctk.CTkTextbox(
            frame,
            corner_radius = 10,
            font = ("Consolas", 12),
            fg_color = "#0a0a0a",
            border_width = 0,
            wrap = "word"
        )
        textbox.pack(expand = True, fill = "both", padx = 15, pady = 15) # Add padding to the textbox.

        # Function to resize the texbox dynamically with the frame.
        def resize_textbox(event):
            textbox.configure(width = event.width, height = event.height)

        # Bind the resizing event to frame adjustments.
        frame.bind("<Configure>", resize_textbox)
        
        # If the metrics file exists, display its content.
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            textbox.insert("0.0", content)
        else: # Else through an error.
            textbox.insert("0.0", f"File not found:\n{txt_path}")

        # Make sure the text box in read-only mode.
        textbox.configure(state = "disabled")