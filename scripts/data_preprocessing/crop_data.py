"""Main script for image cropping operations with manual and automated options.

This script provides a GUI interface for selecting between manual and automated
cropping of H&E images or their corresponding masks. It uses configuration
from YAML files for paths and settings with variable interpolation support.
"""


import os
import argparse
import tkinter as tk
from tkinter import messagebox
import yaml

from . import manual_cropping as MC
from . import automated_cropping as AC
from . import hover_button
from .. import config_helper


class MainGUI(tk.Tk):
    """Main GUI window for selecting cropping operations."""

    def __init__(self, config_data):
        super().__init__()
        self.config_data = config_data  # <-- rename to avoid clash
        self._setup_window()
        self._create_widgets()

    def _setup_window(self):
        self.title("Image Cropping Menu")
        self.geometry("650x440")
        self.resizable(False, False)
        self.config(bg="white")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.destroy()

    def _create_widgets(self):

        top_frame = tk.Frame(self, width=600, height=300, bg="white")
        top_frame.pack(side="top", fill="both")

        menu_frame = tk.Frame(top_frame, width=600, height=300, bg="white")
        menu_frame.pack(side="left", pady=20, fill="both")

        logo_frame = tk.Frame(top_frame, width=600, height=300, bg="white")
        logo_frame.pack(side="right", pady=5, fill="both")

        menu_label = tk.Label(
            menu_frame,
            text="Menu",
            font=("lemon milk", 38, "bold"),
            fg=self.config_data["colors"]["bg"],
            bg="white"
        )
        menu_label.pack(padx=55)

        try:
            logo_photo = tk.PhotoImage(file=self.config_data["paths"]["logo"])
            logo_photo = logo_photo.subsample(2)
            logo_label = tk.Label(logo_frame, image=logo_photo, bg="white")
            logo_label.image = logo_photo
            logo_label.pack(side="right", padx=30)
        except Exception as e:
            print(f"Could not load logo: {e}")

        self._create_decorative_elements()
        self._create_manual_cropping_button()
        self._create_automated_cropping_button()

    def _create_decorative_elements(self):
        tk.Label(self, height=18, width=1, bg=self.config_data["colors"]["bg"]).place(x=44, y=90)
        tk.Label(self, height=18, width=1, bg=self.config_data["colors"]["bg"]).place(x=590, y=90)
        tk.Label(self, height=1, width=200, bg=self.config_data["colors"]["bg"]).place(x=0, y=90)

    def _create_manual_cropping_button(self):
        frame = tk.Frame(
            self,
            width=600,
            height=100,
            bg="white",
            highlightcolor=self.config_data["colors"]["active_bg"],
            highlightthickness=3,
            highlightbackground=self.config_data["colors"]["active_bg"]
        )
        frame.pack(pady=30, ipadx=20, ipady=5)

        btn = hover_button.HoverButton(
            frame,
            text="Manual Cropping",
            activebackground=self.config_data["colors"]["active_bg"],
            command=self._launch_manual_cropping,
            borderwidth=0,
            width=18,
            height=2,
            font=("bold", "15"),
            fg=self.config_data["colors"]["fg"],
            bg=self.config_data["colors"]["bg"]
        )
        btn.pack(side="left", padx=20, pady=10)

        tk.Label(
            frame,
            text="Manually crop your region of interest from\n either H&E images or their corresponding Masks",
            bg="white",
            font=("bold", 9)
        ).pack(side="left", padx=10, pady=10)

    def _create_automated_cropping_button(self):
        frame = tk.Frame(
            self,
            width=600,
            height=100,
            bg="white",
            highlightcolor=self.config_data["colors"]["active_bg"],
            highlightthickness=3,
            highlightbackground=self.config_data["colors"]["active_bg"]
        )
        frame.pack(pady=30, ipadx=13, ipady=5)

        btn = hover_button.HoverButton(
            frame,
            text="Automated Cropping",
            activebackground=self.config_data["colors"]["active_bg"],
            command=self._launch_automated_cropping,
            borderwidth=0,
            width=18,
            height=2,
            font=("bold", "15"),
            fg=self.config_data["colors"]["fg"],
            bg=self.config_data["colors"]["bg"]
        )
        btn.pack(side="left", padx=20, pady=10)

        tk.Label(
            frame,
            text="Automated cropping of either H&E images or Masks\n with coordinates created in the manual cropping",
            bg="white",
            font=("bold", 9)
        ).pack(side="left", padx=10, pady=10)

    def _launch_manual_cropping(self):
        self.destroy()
        MC.ManualCropper(self.config_data)

    def _launch_automated_cropping(self):
        self.destroy()
        AC.AutomatedCropper(self.config_data)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image cropping application for H&E images and masks."
    )
    # Add command line args here if needed
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Get root of project (two levels up from crop_data.py)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    base_path = os.path.join(PROJECT_ROOT, "config", "base_config.yaml")
    dataprep_path = os.path.join(PROJECT_ROOT, "config", "dataprep_config.yaml")

    config = config_helper.ConfigLoader.load_config(base_path, dataprep_path)

    # Verify essential paths exist
    for path_key in ["logo", "arrow_right", "arrow_left"]:
        path = config["paths"].get(path_key)
        if path and not os.path.exists(path):
            print(f"Warning: Path {path_key} ({path}) does not exist")

    app = MainGUI(config)
    app.mainloop()
