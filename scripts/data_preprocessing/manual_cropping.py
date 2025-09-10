#!/usr/bin/env python3
"""Manual image cropping interface for H&E, masks, and IHC images.

This script provides a GUI for manually selecting regions of interest in images
and saving them as separate crops with associated coordinate data.
"""

import os
import pickle
import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox
import urllib.request
from typing import Tuple, List, Optional

import cv2
import numpy as np
import yaml

from . import hover_button
from . import crop_data


class ManualCropper(tk.Tk):
    """Main GUI window for manual image cropping operations."""

    def __init__(self, config_data):
        """Initialize the manual cropping interface.

        Args:
            config: Configuration dictionary loaded from YAML.
        """
        super().__init__()
        self.config_data = config_data  # <-- rename to avoid clash
        self._setup_window()
        self._create_widgets()

        # Cropping state variables
        self.original_images_path = None
        self.square_size = 50  # Initial size of the square
        self.square_pos = [50, 50]  # Initial position [x, y]
        self.roi_id = 1
        self.continue_cropping = True
        self.skip_all_images = False

    def _setup_window(self):
        """Configure the main window properties."""
        self.title("Manual Image Cropping")
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

        tk.Radiobutton(
            rb_frame, text="IHC", value=3, variable=self.image_type,
            bg="white", font=16, command=self._clear_radio_errors
        ).pack(side=tk.LEFT, padx=20)

        # Image selection
        tk.Label(
            self,
            text="Select images you want to crop",
            bg="white",
            font=16
        ).place(x=20, y=140)

        self.file_path_var = tk.StringVar()
        tk.Entry(
            self, textvariable=self.file_path_var, font=16,
            borderwidth=1, width=35, bg="white", relief="solid",
            state="disabled"
        ).place(x=300, y=140)

        hover_button.HoverButton(
            self, text="...", activebackground=self.config_data["colors"]["active_bg"],
            command=self._select_images, borderwidth=0, width=5,
            height=1, font=("bold", 8), fg=self.config_data["colors"]["fg"],
            bg=self.config_data["colors"]["bg"]
        ).place(x=622, y=140)

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

        # Error label
        self.error_label = tk.Label(self, text="", fg="red", bg="white")
        self.error_label.place(x=405, y=162)

    def _clear_radio_errors(self):
        """Clear error styling from radio buttons."""
        for widget in self.winfo_children():
            if isinstance(widget, tk.Radiobutton):
                widget.config(fg="black")

    def _select_images(self):
        """Handle image selection based on chosen type."""
        self.error_label.config(text="")
        self._clear_radio_errors()

        if self.image_type.get() in (1, 2):
            self.original_images_path = filedialog.askopenfilenames(
                title="Select images for cropping",
                filetypes=[("Images", ("*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.png", "*.bmp"))]
            )
        elif self.image_type.get() == 3:
            self.original_images_path = filedialog.askopenfilename(
                title="Select IHC database",
                filetypes=[("Database", ("*.db"))]
            )
        else:
            self.error_label.config(text="Select H&E, Mask or IHC")
            return

        self.file_path_var.set(self.original_images_path)

    def _start_cropping(self):
        """Validate inputs and start the cropping process."""
        if not self.original_images_path:
            self.error_label.config(text="Select valid images")
            return

        if self.image_type.get() == 0:
            for widget in self.winfo_children():
                if isinstance(widget, tk.Radiobutton):
                    widget.config(fg="red")
            return

        # Determine save path based on image type
        if self.image_type.get() == 1:
            save_path = self.config_data["paths"]["roi_images_path"]
        elif self.image_type.get() == 2:
            save_path = self.config_data["paths"]["masks_path"]
        else:  # IHC
            save_path = self.config_data["paths"]["ihc_path"]
            self.original_images_path = self._load_ihc_urls(self.original_images_path)

        # Create directories if they don't exist
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(self.config_data["paths"]["roi_xy_path"], exist_ok=True)

        self.destroy()
        self._process_images(save_path)

    def _load_ihc_urls(self, db_path: str) -> Tuple[str, ...]:
        """Load IHC image URLs from database.

        Args:
            db_path: Path to SQLite database.

        Returns:
            Tuple of image URLs.
        """
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT url FROM elGenes")
        return tuple(row[0] for row in cur.fetchall())

    def _process_images(self, save_path: str):
        for image in self.original_images_path:
            if self.skip_all_images:
                crop_data.MainGUI(self.config_data)

                break  # Skip all remaining images if ESC was pressed earlier

            if not os.path.exists(image) and self.image_type.get() != 3:
                continue

            self.continue_cropping = True
            while self.continue_cropping:
                image_name = os.path.splitext(os.path.basename(image))[0]

                if self.image_type.get() == 3:
                    img = self._load_image_from_url(image)
                else:
                    img = cv2.imread(image, cv2.IMREAD_COLOR)

                if img is None:
                    print(f"Failed to load image: {image}")
                    break

                resized = self._resize_image(img, height=800)

                continue_cropping_image = self._show_cropping_interface(image_name, save_path, img, resized)
                if not continue_cropping_image:
                    break

    def _load_image_from_url(self, url: str) -> Optional[np.ndarray]:
        """Load image from URL.

        Args:
            url: Image URL.

        Returns:
            Loaded image as numpy array or None if failed.
        """
        try:
            with urllib.request.urlopen(url) as req:
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                return cv2.imdecode(arr, -1)
        except Exception as e:
            print(f"Error loading image from URL {url}: {e}")
            return None

    def _show_cropping_interface(self, image_name: str, save_path: str,
                                 original_img: np.ndarray, resized_img: np.ndarray) -> bool:
        """Show the interactive cropping interface.

        Returns:
            bool: True if cropping was completed normally, False if window was closed or ESC pressed.
        """
        while True:
            img_with_square = resized_img.copy()
            cv2.rectangle(
                img_with_square,
                (self.square_pos[0], self.square_pos[1]),
                (self.square_pos[0] + self.square_size, self.square_pos[1] + self.square_size),
                (128, 128, 128), 2
            )

            cv2.imshow("Manual Cropping", img_with_square)
            cv2.setMouseCallback("Manual Cropping", self._handle_mouse)

            key = cv2.waitKey(1)
            if cv2.getWindowProperty("Manual Cropping", cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                return False  # signal that cropping ended by window close

            if key == 27:  # ESC
                cv2.destroyAllWindows()
                self.skip_all_images = True
                return False  # signal that cropping ended by ESC

            elif key == ord("w"):  # Increase size
                self.square_size += 5
            elif key == ord("s") and self.square_size > 5:  # Decrease size
                self.square_size -= 5
            elif ord("1") <= key <= ord("9"):  # Number key pressed
                num_parts = key - ord("0")
                if self._validate_square_position(resized_img):
                    self._process_cropping(num_parts, image_name, save_path, original_img, resized_img)
                    if not messagebox.askyesno(
                            "Continue Cropping",
                            "Do you want to continue cropping in this image?"
                    ):
                        self.continue_cropping = False
                        break
                    cv2.destroyAllWindows()
                    self.square_size = 50

        cv2.destroyAllWindows()
        return True  # signal normal end of cropping this image

    def _validate_square_position(self, img: np.ndarray) -> bool:
        """Validate if square is within image bounds.

        Args:
            img: Image to check against.

        Returns:
            True if square is valid, False otherwise.
        """
        h, w = img.shape[:2]
        return (self.square_pos[0] >= 0 and
                self.square_pos[0] + self.square_size <= w and
                self.square_pos[1] >= 0 and
                self.square_pos[1] + self.square_size <= h)

    def _process_cropping(self, num_parts: int, image_name: str, save_path: str,
                          original_img: np.ndarray, resized_img: np.ndarray):
        """Process the actual cropping operation.

        Args:
            num_parts: Number of parts to divide the square into.
            image_name: Base name of the image.
            save_path: Directory to save cropped images.
            original_img: Original resolution image.
            resized_img: Resized image for display.
        """
        # Convert coordinates from resized to original resolution
        res_h, res_w = resized_img.shape[:2]
        orig_h, orig_w = original_img.shape[:2]

        # Convert square position and size
        x_ratio = orig_w / res_w
        y_ratio = orig_h / res_h

        orig_x = int(self.square_pos[0] * x_ratio)
        orig_y = int(self.square_pos[1] * y_ratio)
        orig_size = int(self.square_size * x_ratio)  # Assuming square aspect ratio

        part_size = orig_size // num_parts

        # Crop and save each part
        for i in range(num_parts):
            for j in range(num_parts):
                x = orig_x + j * part_size
                y = orig_y + i * part_size

                crop = original_img[y:y + part_size, x:x + part_size]
                crop_path = os.path.join(save_path, f"{image_name}_{self.roi_id}.png")
                cv2.imwrite(crop_path, crop)

                # Save coordinates
                coord_path = os.path.join(
                    self.config_data["paths"]["roi_xy_path"],
                    f"{image_name}_{self.roi_id}.pckl"
                )
                with open(coord_path, "wb") as f:
                    pickle.dump(([x, x + part_size], [y, y + part_size]), f)

                self.roi_id += 1

    def _handle_mouse(self, event, x, y, flags, param):
        """Handle mouse events for square positioning."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.square_pos = [x - self.square_size // 2, y - self.square_size // 2]

    def _resize_image(self, image: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
        """Resize image while maintaining aspect ratio.

        Args:
            image: Input image.
            width: Target width.
            height: Target height.

        Returns:
            Resized image.
        """
        h, w = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            ratio = height / h
            dim = (int(w * ratio), height)
        else:
            ratio = width / w
            dim = (width, int(h * ratio))

        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def _return_to_main(self):
        """Return to the main menu."""
        self.original_images_path = None
        self.destroy()
        crop_data.MainGUI(self.config_data)


if __name__ == "__main__":
    # Get root of project (two levels up from crop_data.py)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    base_path = os.path.join(PROJECT_ROOT, "config", "base_config.yaml")
    dataprep_path = os.path.join(PROJECT_ROOT, "config", "dataprep_config.yaml")

    config = config_helper.ConfigLoader.load_config(base_path, dataprep_path)

    app = ManualCropper(config)
    app.mainloop()