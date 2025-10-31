# Import dependencies.
import customtkinter as ctk # Import 'CustomTkinter' for modern looking tkinter GUI.
from gui.services.model_service import ModelService # Import 'ModelService' class for type hinting.
from customtkinter import CTkImage # Import 'CTkImage' to display image in CTk widget.
from tkinter import filedialog # Import 'filedialog' for file chooser dialog.
from PIL import Image # Import 'Image' to handle image loading and resizing.



# Class to handle the upload image classfication.
class UploadTab(ctk.CTkFrame):

    # Initialise the upload tab.
    def __init__(self, parent : ctk.CTkBaseClass, model_service : ModelService) -> None:
        
        # Initialise the CTk Frame.
        super().__init__(parent, fg_color = "transparent")

        # Make sure the frame fill all the space available.
        self.pack(fill = "both", expand = True)

        # Store a reference of the model service.
        self.model_service = model_service

        # Create a container frame inside this tab for layout organisation.
        self.container = ctk.CTkFrame(self, fg_color = "transparent")
        self.container.pack(fill = "both", expand = True, padx =20) # Make sure organising frame to take all the space avaliable and padding.

        # Create a button that allow the picking of an image.
        self.choose_btn = ctk.CTkButton(
            self.container,
            text = "Choose Image",
            command = self.choose_image,
            font = ("Segoe UI", 16, "bold"),
            height = 50,
            corner_radius = 10,
            hover_color = "#1a5a8e",
            fg_color = "#1f6aa5"
        )
        self.choose_btn.pack(pady=(10, 20)) # Add padding to the upload button.

        # Create a frame that will display the image.
        self.image_frame = ctk.CTkFrame(
            self.container,
            corner_radius = 15,
            border_width = 2,
            border_color = "#3b3b3b",
            fg_color = "#2b2b2b"
        )
        self.image_frame.pack(padx = 20, fill = "both", expand = False) # Add padding to the image frame and make sure it fills all avaliable page.

        # Create a label where the seleted image will be displayed.
        self.image_label = ctk.CTkLabel(
            self.image_frame,
            text = "",
            font = ("Segoe UI", 14)
        )
        self.image_label.pack(pady = 30, expand = False) # Add padding to the image label.

        # Create a frame what will display the prediction.
        self.result_frame = ctk.CTkFrame(
            self.container,
            corner_radius = 10,
            fg_color = "#2b2b2b",
            border_width = 2,
            border_color = "#1f6aa5"
        )
        self.result_frame.pack(pady = (10, 20), padx = 20, fill = "x") # Add padding the prediction frame.

        # Label to display the prediction.
        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text = "Prediction: Awaiting Image",
            font = ("Segoe UI", 16, "bold"),
            text_color = "#b0b0b0"
        )
        self.result_label.pack(pady = 20) # Add a padding to the prediction label.


    # Function to handle image selection and prediction.
    def choose_image(self) -> None:

        # Open a dialog to choose an image.
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])

        # If there is no file selected, exit the function.
        if not file_path:
            return

        # Load the image and resize it.
        img = Image.open(file_path)
        img = img.resize((250, 250))

        # Convert the PIL image into a CTk image.
        img_ctk = CTkImage(light_image = img, dark_image = img, size=(300, 300))

        # Display the image into the image label.
        self.image_label.configure(image = img_ctk, text = "")
        self.image_label.image = img_ctk

        # Change the Upload button text.
        self.choose_btn.configure(text = "Upload Another Image")

        # Use the model to predict image class and confidance.
        pred_class, pred_conf = self.model_service.predict(file_path)

        # Get the text colour base on the object class and confidance.
        if pred_conf > 0.7:
            color = "#4ade80"
        else:
            color = "#f87171"

        # Format the prediction text.
        display_text = f"Prediction: {pred_class}\n Confidence: {pred_conf:.1%}"
        
        # Update the prediction label with the prediction colour and text.
        self.result_label.configure(
            text = display_text,
            font = ("Segoe UI", 16, "bold"),
            text_color = color,
            justify = "center"
        )