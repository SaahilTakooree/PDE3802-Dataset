# gui_utils.py
import tkinter as tk

class WindowConfig:
    DEFAULT_WIDTH = 900
    DEFAULT_HEIGHT = 600

    def configure_window(window: tk.Toplevel | tk.Tk, width=None, height=None):
        # Set the window size and center it on the screen.
        width = width or WindowConfig.DEFAULT_WIDTH
        height = height or WindowConfig.DEFAULT_HEIGHT
        
        # Get screen width and height
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        
        # Calculate position x, y to center the window
        x = int((screen_width - width) / 2)
        y = int((screen_height - height) / 2)
        
        # Set geometry
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.resizable(False, False)