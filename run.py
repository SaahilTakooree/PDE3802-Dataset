from gui.main_app import MainApp
from gui.model_service import ModelService
import customtkinter as ctk

if __name__ == "__main__":
    # appearance (you can change to "dark" if you want)
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")

    model = ModelService(model_path=r"C:\Users\mehar\Downloads\PDE3802-Dataset\office_supplies_classifier\train_v4_final\weights\best.pt")  # or "best.pt"
    app = MainApp(model_service=model)
    app.run()
