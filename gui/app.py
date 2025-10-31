# Import dependencies.
from gui.services.model_service import ModelService # Import 'ModelService' class for type hinting.
import customtkinter as ctk # Import 'CustomTkinter' for a modern verion of Tkinter to create a GUI.
from gui.views.main_menu import MainMenu # Import the Main_Menu for the GUI.



# Application class to handle the main GUI window.
class Application:

    # Initialise the app.
    def __init__ (self, model_service : ModelService) -> None:

        # Store the model service.
        self.model_service = model_service

        # Create the main CustomTkinter window.
        self.root = ctk.CTk()

        # Set the title for the window.
        self.root.title("Office Item Classifier")

        # Set the appearence and colour theme for the window.
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Set the width and height of the window.
        width, height = 1000, 700
        
        # Center the window on the screen.
        self.center_window(width, height)

        # Prevent window resizing.
        self.root.resizable(False, False)

        # Initialise the main menu.
        self.main_menu = MainMenu(self.root, self.model_service)


    # Function to center the window to the center of the screen.
    def center_window(self, width : int, height : int) -> None:
        
        # Update the window to make sure the winfo_ method returns the correct value.
        self.root.update_idletasks()
        
        # Get the window width.
        screen_width = self.root.winfo_screenwidth()
        
        # Get the window height.
        screen_height = self.root.winfo_screenheight()

        # Calculate the x and y position to  center the window.
        x = int((screen_width // 2) - (width // 2))
        y = int((screen_height // 2) - (height // 2))

        # Set the window geometry with position.
        self.root.geometry(f"{width}x{height}+{x}+{y}")


    # Starting the main load the dieplayloo  the window
    def run(self) -> None:
        self.root.mainloop()