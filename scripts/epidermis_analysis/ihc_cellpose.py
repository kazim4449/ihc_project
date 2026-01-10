"""
Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021).
Cellpose: a generalist algorithm for cellular segmentation.
Nature Methods, 18(1), 100–106.

Di Cataldo, S., Ficarra, E., Acquaviva, A., & Macii, E. (2010).
Automated segmentation of tissue images for computerized IHC analysis.
https://doi.org/10.1016/j.cmpb.2010.02.002
"""

import os
import json
import argparse
import sqlite3
import urllib.request
from datetime import datetime

import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.color import rgb2hed, hed2rgb
from skimage import img_as_ubyte

from cellpose import models

from .. import config_helper

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def load_last_processed_id(path: str) -> int:
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0


def save_last_processed_id(path: str, value: int) -> None:
    with open(path, "w") as f:
        f.write(str(value))


def download_image(url: str) -> np.ndarray:
    with urllib.request.urlopen(url) as r:
        arr = np.asarray(bytearray(r.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

class IHC_Cellpose:
    """
    Cellpose-based IHC cell segmentation pipeline.
    """

    def __init__(self, config: dict, version: str, masks_ihc_path: str):
        self.config = config
        self.version = version
        self.masks_ihc_path = masks_ihc_path

        data_root = config["paths"]["data_root"]

        # Versioned outputs
        self.output_root = os.path.join(data_root, version, "cellpose")
        self.masks_out = os.path.join(self.output_root, "outputs_masks")
        self.vis_out = os.path.join(self.output_root, "outputs_visualization")

        os.makedirs(self.masks_out, exist_ok=True)
        os.makedirs(self.vis_out, exist_ok=True)

        self.last_id_file = os.path.join(
            self.output_root, "last_processed_id.txt"
        )

        self.db_path = config["paths"]["database_path"]

        # Cellpose model
        self.model = models.Cellpose(
            gpu=True,
            model_type="cyto2"
        )
        self.channels = [0, 0]

        self.run_metadata_path = os.path.join(
            self.output_root, "run_metadata.json"
        )

        self.run_metadata = {
            "pipeline": "IHC_Cellpose",
            "version": self.version,
            "created_at": datetime.now().isoformat(),

            # Environment
            #"user": getpass.getuser(),
            #"hostname": socket.gethostname(),

            # Inputs
            "masks_ihc_path": self.masks_ihc_path,
            "database_path": self.db_path,

            # Model
            "model": {
                "framework": "cellpose",
                "model_type": "cyto2",
                "gpu": True,
                "channels": self.channels,
                "diameter": None
            },

            # Preprocessing summary
            "preprocessing": [
                "RGB → HED",
                "hematoxylin extraction",
                "invert",
                "histogram equalization",
                "epidermis-only masking",
                "crop to tissue"
            ],

            # Outputs
            "outputs": {
                "output_root": self.output_root,
                "masks_out": self.masks_out,
                "visualizations_out": self.vis_out
            },

            # Config provenance
            "config": {
                "paths": self.config.get("paths", {}),
                #"hash": hash_dict(self.config)
            },

            # Runtime stats
            "stats": {
                "processed": 0,
                "failed": 0
            }
        }

        with open(self.run_metadata_path, "w") as f:
            json.dump(self.run_metadata, f, indent=2)

    # -------------------------------------------------------------------------
    @staticmethod
    def extract_epidermis(image: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
        """
        Return an image where only the epidermis (mask==2) is kept,
        and all other pixels are white.
        """
        epidermis_only = np.copy(image)
        epidermis_only[tissue_mask != 2] = 255  # white-out non-epidermis
        return epidermis_only

    @staticmethod
    def crop_to_tissue(image: np.ndarray, tissue_mask: np.ndarray):
        """
        Crop the image to the bounding box of the tissue (mask==2, epidermis).
        Returns the cropped image and bounding box (x, y, w, h).
        """
        mask = (tissue_mask == 2).astype(np.uint8)  # epidermis only
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, None

        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y:y + h, x:x + w], (x, y, w, h)

    @staticmethod
    def prepare_H_channel(epidermis_rgb: np.ndarray, return_raw: bool = False):
        """
        Prepare the Hematoxylin (H) channel for model input.
        Returns both the raw H channel (after hed2rgb) and the final H_eq.
        """
        # Convert RGB to HED
        hed = rgb2hed(epidermis_rgb)

        # Hematoxylin channel (old pipeline uses channel 0)
        null = np.zeros_like(hed[:, :, 0])
        H_rgb_like = hed2rgb(np.stack((hed[:, :, 0], null, null), axis=-1))

        # Convert to 0-255 ubyte
        H = img_as_ubyte(H_rgb_like)

        # Take single channel for inversion + equalization
        H_gray = H[:, :, 0]  # channel 0 matches old pipeline
        H_inv = np.invert(H_gray)
        H_eq = cv2.equalizeHist(H_inv)

        if return_raw:
            return H_gray, H_eq
        else:
            return H_eq

    # -------------------------------------------------------------------------

    def save_visualization(
            self, original, tissue_mask, cropped, epidermis_only,
            H_raw, H_eq, masks, overlay_name, gene_name, mainID, mask_base
    ):
        """
        Save multi-panel figure including:
        - HED separation and final H channel
        """
        plt.figure(figsize=(24, 4))
        plt.suptitle(f"{gene_name} | ID: {mainID}", fontsize=16)

        overlay = original.copy()
        overlay[masks > 0] = [255, 0, 0]  # red overlay

        panels = [original, tissue_mask, cropped, epidermis_only, H_raw, H_eq, masks, overlay]
        titles = [
            "Original Image", "Tissue Mask", "Cropped Tissue",
            "Epidermis Only", "H Channel (raw)", "H Channel EQ",
            "Predicted Mask", "Overlay Mask"
        ]

        for i, panel in enumerate(panels):
            plt.subplot(1, len(panels), i + 1)
            plt.title(titles[i])
            if panel.ndim == 2:
                plt.imshow(panel, cmap="gray")
            else:
                plt.imshow(panel)
            plt.axis("off")

        os.makedirs(self.vis_out, exist_ok=True)
        plt.savefig(
            os.path.join(self.vis_out, f"{mask_base}.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

    def process_row(self, main_id: int, name: str, url: str):
        print(f"[{main_id}] {name}")

        mask_base = f"{url.split('/')[-2]}-{url.split('/')[-1].split('.')[0]}"
        mask_path = os.path.join(self.masks_ihc_path, f"{mask_base}.npz")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing IHC mask: {mask_path}")

        tissue_mask = np.load(mask_path)["mask"]

        image = download_image(url)

        cropped, bbox = self.crop_to_tissue(image, tissue_mask)
        if cropped is None:
            raise RuntimeError("No tissue region detected")

        # Extract epidermis from cropped tissue
        x, y, w, h = bbox
        tissue_crop_mask = tissue_mask[y:y + h, x:x + w]
        epidermis_only = self.extract_epidermis(cropped, tissue_crop_mask)

        # Prepare H channel
        # We'll split into raw H (after hed2rgb) and final H_eq
        H_raw, H_eq = self.prepare_H_channel(epidermis_only, return_raw=True)

        # Model prediction
        masks, _, _, _ = self.model.eval(
            H_eq,
            diameter=None,
            channels=[0, 0]
        )

        full_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        full_mask[y:y + h, x:x + w] = masks

        np.savez_compressed(
            os.path.join(self.masks_out, f"{mask_base}.npz"),
            mask=full_mask,
            crop_bbox=bbox

        )

        # Save visualization
        self.save_visualization(
            original=image,
            tissue_mask=tissue_crop_mask,
            cropped=cropped,
            epidermis_only=epidermis_only,
            H_raw=H_raw,
            H_eq=H_eq,
            masks=full_mask,
            overlay_name=None,  # optional
            gene_name=name,
            mainID=main_id,
            mask_base=mask_base
        )

    # -------------------------------------------------------------------------

    def run(self):
        last_id = load_last_processed_id(self.last_id_file)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            "SELECT mainID, name, url FROM elGenes "
            "WHERE mainID > ? ORDER BY mainID",
            (last_id,)
        )
        rows = cur.fetchall()
        conn.close()

        for main_id, name, url in rows:
            try:
                self.process_row(main_id, name, url)
                save_last_processed_id(self.last_id_file, main_id)
                self.run_metadata["stats"]["processed"] += 1
            except Exception as e:
                self.run_metadata["stats"]["failed"] += 1
                print(f"[ERROR] {name}: {e}")

        self.run_metadata["finished_at"] = datetime.now().isoformat()
        self.run_metadata["last_processed_id"] = (
            rows[-1][0] if rows else last_id
        )

        with open(self.run_metadata_path, "w") as f:
            json.dump(self.run_metadata, f, indent=2)


# -----------------------------------------------------------------------------
# CLI entrypoint (module-safe)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_cfg = os.path.join(PROJECT_ROOT, "config", "base_config.yaml")
    pred_cfg = os.path.join(PROJECT_ROOT, "config", "prediction_config.yaml")

    config = config_helper.ConfigLoader.load_config(base_cfg, pred_cfg)

    parser = argparse.ArgumentParser("IHC Cellpose segmentation")
    parser.add_argument(
        "--version",
        required=True,
        help="Dataset / model version (e.g. v001)"
    )
    parser.add_argument(
        "--masks_ihc_path",
        required=True,
        help="Path to IHC tissue masks (.npz) from prediction pipeline"
    )

    args = parser.parse_args()

    pipeline = IHC_Cellpose(
        config=config,
        version=args.version,
        masks_ihc_path=args.masks_ihc_path
    )
    pipeline.run()
