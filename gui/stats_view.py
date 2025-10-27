# gui/stats_view.py
import customtkinter as ctk
from PIL import Image, ImageTk
import os

ASSETS_DIR = os.path.join(os.getcwd(), "assets")

class StatsViewFrame(ctk.CTkFrame):
    def __init__(self, parent, on_back):
        super().__init__(parent)
        self.on_back = on_back

        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=12)

        ctk.CTkButton(btn_frame, text="Confusion Matrix", command=self.load_confusion_matrix).pack(side="left", padx=6)
        ctk.CTkButton(btn_frame, text="Epoch Accuracy & Loss", command=self.load_accuracy_loss).pack(side="left", padx=6)
        ctk.CTkButton(btn_frame, text="Metrics (text)", command=self.load_metrics).pack(side="left", padx=6)
        ctk.CTkButton(btn_frame, text="Back", command=self.on_back).pack(side="left", padx=6)

        self.display_area = ctk.CTkLabel(self, text="")
        self.display_area.pack(pady=12, expand=True)

        self.text_widget = ctk.CTkTextbox(self, width=800, height=300)
        self.text_widget.pack_forget()

    def load_confusion_matrix(self):
        path = os.path.join(ASSETS_DIR, "confusion_matrix.png")
        if os.path.exists(path):
            img = Image.open(path).resize((800, 400))
            tkimg = ImageTk.PhotoImage(img)
            self.display_area.configure(image=tkimg, text="")
            self.display_area.image = tkimg
            self.text_widget.pack_forget()
        else:
            self.display_area.configure(text="Confusion matrix not found.", image="")
            self.text_widget.pack_forget()

    def load_accuracy_loss(self):
        path = os.path.join(ASSETS_DIR, "epoch_accuracy&loss.png")
        if os.path.exists(path):
            img = Image.open(path).resize((800, 400))
            tkimg = ImageTk.PhotoImage(img)
            self.display_area.configure(image=tkimg, text="")
            self.display_area.image = tkimg
            self.text_widget.pack_forget()
        else:
            self.display_area.configure(text="Epoch accuracy & loss not found.", image="")
            self.text_widget.pack_forget()

    def load_metrics(self):
        path = os.path.join(ASSETS_DIR, "metrics.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8-sig") as f:
                text = f.read()
            self.text_widget.delete("0.0", "end")
            self.text_widget.insert("0.0", text)
            self.text_widget.pack(pady=12)
            self.display_area.configure(image="", text="")
        else:
            self.display_area.configure(text="Metrics file not found.", image="")
            self.text_widget.pack_forget()
