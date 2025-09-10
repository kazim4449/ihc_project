#!/usr/bin/env python3
"""Automated image cropping interface for H&E and mask images.

This script provides a GUI for automatically cropping images using pre-defined
coordinate files and saving the cropped regions.
"""

import os
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

from . import hover_button
from . import crop_data


class AutomatedCropper(tk.Tk):
    """Main GUI window for automated image cropping operations."""

    def __init__(self, config_data):
        """Initialize the automated cropping interface.

        Args:
            config: Configuration dictionary loaded from YAML.
        """
        super().__init__()
        self.config_data = config_data  # <-- rename to avoid clash
        self._setup_window()
        self._create_widgets()

        # State variables
        self.original_images_path = None
        self.cropping_co_path = None

    def _setup_window(self):
        """Configure the main window properties."""
        self.title("Automated Image Cropping")
        self.geometry("700x300")
        self.resizable(False, False)
        self.config(bg="white")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        """Handle window closing event."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.destroy()
            crop_data.MainGUI(self.config_data)

    def _create_widgets(self):
        """Create and arrange all GUI widgets."""
        # Image type selection
        tk.Label(
            self,
            text="Do you want to crop\nH&E images or Masks?",
            bg="white",
            font=16,
            justify=tk.LEFT
        ).place(x=20, y=40)

        self.image_type = tk.IntVar()

        rb_frame = tk.Frame(self, bg="white")
        rb_frame.place(x=300, y=40)

        tk.Radiobutton(
            rb_frame, text="H&E", value=1, variable=self.image_type,
            bg="white", font=16, command=self._clear_radio_errors
        ).pack(side=tk.LEFT, padx=20)

        tk.Radiobutton(
            rb_frame, text="Mask", value=2, variable=self.image_type,
            bg="white", font=16, command=self._clear_radio_errors
        ).pack(side=tk.LEFT, padx=20)

        # Image selection
        tk.Label(
            self,
            text="Select images you want to crop",
            bg="white",
            font=16
        ).place(x=20, y=120)

        self.image_path_var = tk.StringVar()
        tk.Entry(
            self, textvariable=self.image_path_var, font=16,
            borderwidth=1, width=35, bg="white", relief="solid",
            state="disabled"
        ).place(x=300, y=120)

        hover_button.HoverButton(
            self, text="...", activebackground=self.config_data["colors"]["active_bg"],
            command=self._select_images, borderwidth=0, width=5,
            height=1, font=("bold", 8), fg=self.config_data["colors"]["fg"],
            bg=self.config_data["colors"]["bg"]
        ).place(x=622, y=120)

        # Coordinate file selection
        tk.Label(
            self,
            text="Select cropping coordinates",
            bg="white",
            font=16
        ).place(x=20, y=185)

        self.coord_path_var = tk.StringVar()
        tk.Entry(
            self, textvariable=self.coord_path_var, font=16,
            borderwidth=1, width=35, bg="white", relief="solid",
            state="disabled"
        ).place(x=300, y=185)

        hover_button.HoverButton(
            self, text="...", activebackground=self.config_data["colors"]["active_bg"],
            command=self._select_coordinates, borderwidth=0, width=5,
            height=1, font=("bold", 8), fg=self.config_data["colors"]["fg"],
            bg=self.config_data["colors"]["bg"]
        ).place(x=622, y=185)

        # Action buttons
        hover_button.HoverButton(
            self, text="Start", activebackground=self.config_data["colors"]["active_bg"],
            command=self._start_cropping, borderwidth=0, width=13,
            height=1, font=("bold", 10), fg=self.config_data["colors"]["fg"],
            bg=self.config_data["colors"]["bg"]
        ).place(x=350, y=260)

        hover_button.HoverButton(
            self, text="Back", activebackground=self.config_data["colors"]["active_bg"],
            command=self._return_to_main, borderwidth=0, width=13,
            height=1, font=("bold", 10), fg=self.config_data["colors"]["fg"],
            bg=self.config_data["colors"]["bg"]
        ).place(x=200, y=260)

        # Error labels
        self.image_error = tk.Label(self, text="", fg="red", bg="white")
        self.image_error.place(x=405, y=142)

        self.coord_error = tk.Label(self, text="", fg="red", bg="white")
        self.coord_error.place(x=405, y=207)

    def _clear_radio_errors(self):
        """Clear error styling from radio buttons."""
        for widget in self.winfo_children():
            if isinstance(widget, tk.Radiobutton):
                widget.config(fg="black")

    def _select_images(self):
        """Handle image selection."""
        self.image_error.config(text="")
        self.original_images_path = filedialog.askopenfilenames(
            title="Select images for cropping",
            filetypes=[("Images", ("*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.png", "*.bmp"))]
        )
        self.image_path_var.set(self.original_images_path)

    def _select_coordinates(self):
        """Handle coordinate file selection."""
        self.coord_error.config(text="")
        self.cropping_co_path = filedialog.askopenfilenames(
            title="Select cropping coordinates",
            filetypes=[("Pickle files", "*.pckl")],
            initialdir=self.config_data["paths"]["roi_xy_path"]
        )
        self.coord_path_var.set(self.cropping_co_path)

    def _start_cropping(self):
        """Validate inputs and start the cropping process."""
        if not self.original_images_path:
            self.image_error.config(text="Select valid images")
            return

        if not self.cropping_co_path:
            self.coord_error.config(text="Select valid coordinate files")
            return

        if self.image_type.get() == 0:
            for widget in self.winfo_children():
                if isinstance(widget, tk.Radiobutton):
                    widget.config(fg="red")
            return

        # Determine save path based on image type
        save_path = (
            self.config_data["paths"]["roi_images_path"]
            if self.image_type.get() == 1
            else self.config_data["paths"]["masks_path"]
        )

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        self.destroy()
        self._process_cropping(save_path)

    def _process_cropping(self, save_path: str):
        """Process all selected images with their coordinates.

        Args:
            save_path: Directory to save cropped images.
        """
        # Create mapping of base names to coordinate files
        coord_map = {}
        for coord_file in self.cropping_co_path:
            if not os.path.exists(coord_file):
                continue

            base_name = os.path.splitext(os.path.basename(coord_file))[0]
            base_name = base_name.rsplit('_', 1)[0]  # Remove ROI ID
            coord_map.setdefault(base_name, []).append(coord_file)

        # Process each image
        for image_path in self.original_images_path:
            if not os.path.exists(image_path):
                continue

            image_name = os.path.splitext(os.path.basename(image_path))[0]

            if image_name not in coord_map:
                continue

            # Load the image once
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            # Process all coordinates for this image
            for coord_file in coord_map[image_name]:
                self._process_single_crop(img, image_name, coord_file, save_path)

        # Return to the automated cropping interface
        AutomatedCropper(self.config_data).mainloop()

    def _process_single_crop(self, img: np.ndarray, image_name: str,
                             coord_file: str, save_path: str):
        """Process a single crop operation.

        Args:
            img: Original image data.
            image_name: Base name of the image.
            coord_file: Path to coordinate file.
            save_path: Directory to save cropped image.
        """
        try:
            with open(coord_file, 'rb') as f:
                all_x, all_y = pickle.load(f)

            # Extract ROI ID from coordinate filename
            coord_name = os.path.splitext(os.path.basename(coord_file))[0]
            roi_id = coord_name.rsplit('_', 1)[1]

            # Crop and save the image
            roi = img[all_y[0]:all_y[1], all_x[0]:all_x[1]]
            output_path = os.path.join(save_path, f"{image_name}_{roi_id}.png")
            cv2.imwrite(output_path, roi)

        except (pickle.PickleError, IOError) as e:
            print(f"Error processing {coord_file}: {e}")

    def _return_to_main(self):
        """Return to the main menu."""
        self.destroy()
        crop_data.MainGUI(self.config_data)


if __name__ == "__main__":
    # Get root of project (two levels up from crop_data.py)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    base_path = os.path.join(PROJECT_ROOT, "config", "base_config.yaml")
    dataprep_path = os.path.join(PROJECT_ROOT, "config", "dataprep_config.yaml")

    config = config_helper.ConfigLoader.load_config(base_path, dataprep_path)

    app = AutomatedCropper(config)
    app.mainloop()