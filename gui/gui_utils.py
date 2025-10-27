import customtkinter as ctk

class WindowConfig:
    DEFAULT_WIDTH = 1000
    DEFAULT_HEIGHT = 700

    @staticmethod
    def configure_window(root: ctk.CTk, width=None, height=None):
        width = width or WindowConfig.DEFAULT_WIDTH
        height = height or WindowConfig.DEFAULT_HEIGHT
        root.geometry(f"{width}x{height}")
        root.resizable(False, False)