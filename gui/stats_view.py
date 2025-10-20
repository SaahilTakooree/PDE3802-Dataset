# Import all dependencies.
import os
import tkinter as tk
from PIL import Image, ImageTk
from gui_utils import WindowConfig



# Define a class to create a Stats View window to view the stats of the model.
class StatsView:
    
    # Initialise the StatsView window.
    def __init__(self, root : tk.Tk):
        self.root = root
        
        # Create a separate top-level window to view the model stats.
        self.window = tk.Toplevel(root)
        self.window.title("Office Item Classifier - View Model Stats")
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)
        
        # Configure size & center.
        WindowConfig.configure_window(self.window)
        
        # Get the directory of the current script.
        assets_dir = os.path.join(os.getcwd(), "assets")

        # Get assets directory.
        self.assets_dir = os.path.join(os.getcwd(), "assets")

        # Create a frame for the buttons.
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=5)

        # Button to load the confusion metrix.
        confusion_metrix_button = tk.Button(button_frame, text="Confusion Matrix", command=self.load_confusion_matrix)
        confusion_metrix_button.pack(side=tk.LEFT, padx=5)
        
        # Button to load the epoch accuracy and loss.
        epoch_accuracy_losst_button = tk.Button(button_frame, text="Epoch Accuracy & Loss", command=self.load_accuracy_loss)
        epoch_accuracy_losst_button.pack(side=tk.LEFT, padx=5)
        
        # Button to laod the metrics.
        metrics_button = tk.Button(button_frame, text="Metrics", command=self.load_metrics)
        metrics_button.pack(side=tk.LEFT, padx=5)
        
        # Display frame.
        self.display_frame = tk.Frame(self.window)
        self.display_frame.pack(pady=10)

        # Image area.
        self.image_label = tk.Label(self.display_frame)
        self.image_label.pack()

        # Text area for metrics.
        self.text_widget = tk.Text(self.display_frame, height=20, width=80)
        self.text_widget.pack_forget()

        # Back button.
        tk.Button(self.window, text="Back", command=self.close_window).pack(pady=5)


    # Load the confusion matrix.
    def load_confusion_matrix(self):
        # Hide text if visible.
        self.text_widget.pack_forget()
        try:
            path = os.path.join(self.assets_dir, "confusion_matrix.png")
            image = Image.open(path)
            image = image.resize((800, 400))
            self.image_tk = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.image_tk, text='')
            self.image_label.pack()
        except Exception:
            self.image_label.config(text="Confusion matrix not found.", image='')
            self.image_label.pack()

    # Load the epoch accuracy and loss.
    def load_accuracy_loss(self):
        self.text_widget.pack_forget()
        try:
            path = os.path.join(self.assets_dir, "epoch_accuracy&loss.png")
            image = Image.open(path)
            image = image.resize((800, 400))
            self.image_tk = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.image_tk, text='')
            self.image_label.pack()
        except Exception:
            self.image_label.config(text="Epoch accuracy & loss not found.", image='')
            self.image_label.pack()

    
    # load the metrics.
    def load_metrics(self):
        # Hide image
        self.image_label.pack_forget()
        try:
            path = os.path.join(self.assets_dir, "metrics.txt")
            with open(path, "r", encoding="utf-8-sig") as f:
                metrics_text = f.read()
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert(tk.END, metrics_text)
            self.text_widget.config(state=tk.DISABLED)
            self.text_widget.pack()
        except Exception:
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert(tk.END, "Metrics file not found.")
            self.text_widget.config(state=tk.DISABLED)
            self.text_widget.pack()


    # Close the Tkinter window.
    def close_window(self):
        self.window.destroy()
