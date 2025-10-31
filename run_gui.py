# Import dependencies.
import os # Import 'os' to get system specific function.
import sys # Import 'sys' to get file path and directory handling function.
from gui.services.model_service import ModelService # Import the model handling service class.
from gui.app import Application # Import the main GUI application class.



# Main entry point for the application. It initialise the model service and start the GUI.
if __name__ == "__main__":

    # Get the model file location.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "office_supplies_classifier", "train_v4_final", "weights", "best.pt")

    # Check if the model exists.
    if not os.path.exists(model_path):
        print("Unable to continue as unable to find model at specify path. Please check model path.") # Explain why the program has stop in the terminal
        sys.exit(1) # If model is missing, exit the program.
    else:
        model_service = ModelService(model_path) # Load the model into memory.

        # Create and run application.
        app = Application(model_service)
        app.run()