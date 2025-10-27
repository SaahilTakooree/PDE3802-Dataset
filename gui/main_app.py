# gui/main_app.py
import customtkinter as ctk
from .gui_utils import WindowConfig
from .camera_view import CameraViewFrame
from .upload_view import UploadViewFrame
from .stats_view import StatsViewFrame

class MainApp:
    """
    Single-window application that switches between frames:
    - Home (menu)
    - Camera view
    - Upload view
    - Stats view
    """

    def __init__(self, model_service):
        self.model_service = model_service
        self.root = ctk.CTk()
        self.root.title("Office Item Classifier")
        WindowConfig.configure_window(self.root)

        # layout frames container
        self.container = ctk.CTkFrame(self.root)
        self.container.pack(fill="both", expand=True, padx=12, pady=12)

        # top app header
        header = ctk.CTkLabel(self.root, text="Office Item Classifier", font=ctk.CTkFont(size=20, weight="bold"))
        header.place(x=20, y=8)

        # frames dictionary
        self.frames = {}

        # create frames
        self.frames["menu"] = self._create_menu()
        self.frames["camera"] = CameraViewFrame(self.container, self.model_service, on_back=self.show_menu)
        self.frames["upload"] = UploadViewFrame(self.container, self.model_service, on_back=self.show_menu)
        self.frames["stats"] = StatsViewFrame(self.container, on_back=self.show_menu)

        # show menu by default
        self.show_menu()

    def _create_menu(self):
        frame = ctk.CTkFrame(self.container)
        # Buttons in a centered column card
        card = ctk.CTkFrame(frame, corner_radius=12, width=420, height=360)
        card.place(relx=0.5, rely=0.5, anchor="center")


        # Title
        ctk.CTkLabel(card, text="Main Menu", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20,10))

        btn_camera = ctk.CTkButton(card, text="Live Camera Detection", width=300, height=40, command=lambda: self.show_frame("camera"))
        btn_camera.pack(pady=10)

        btn_upload = ctk.CTkButton(card, text="Upload Image Detection", width=300, height=40, command=lambda: self.show_frame("upload"))
        btn_upload.pack(pady=10)

        btn_stats = ctk.CTkButton(card, text="View Model Stats", width=300, height=40, command=lambda: self.show_frame("stats"))
        btn_stats.pack(pady=10)

        btn_quit = ctk.CTkButton(card, text="Quit", width=300, height=40, command=self.root.quit)
        btn_quit.pack(pady=10)

        return frame

    def show_frame(self, name):
        # hide all frames then pack the requested one
        for f in self.frames.values():
            try:
                f.pack_forget()
            except Exception:
                pass
        frame = self.frames[name]
        frame.pack(fill="both", expand=True)

        # If camera frame, start camera loop
        if name == "camera":
            frame.start_capture()
        else:
            # if leaving camera, ensure camera stopped
            cam = self.frames.get("camera")
            if cam:
                cam.stop_capture()

    def show_menu(self):
        self.show_frame("menu")

    def run(self):
        self.root.mainloop()
