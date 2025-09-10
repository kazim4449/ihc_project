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
from PIL import Image, ImageTk
import cv2


class HoverButton(tk.Button):
	"""Custom button class with hover effects.

    Attributes:
        defaultBackground (str): The default background color of the button.
    """

	def __init__(self, master, **kw):
		"""Initializes a HoverButton with hover effects.

        Args:
            master: Parent widget.
            **kw: Additional keyword arguments for tk.Button.
        """
		tk.Button.__init__(self, master=master, **kw)
		self.defaultBackground = self["background"]
		self.bind("<Enter>", self.on_enter)
		self.bind("<Leave>", self.on_leave)

	def on_enter(self, event):
		"""Changes button background when mouse enters.

        Args:
            event: Tkinter event object.
        """
		self["background"] = self["activebackground"]

	def on_leave(self, event):
		"""Resets button background when mouse leaves.

        Args:
            event: Tkinter event object.
        """
		self["background"] = self.defaultBackground
