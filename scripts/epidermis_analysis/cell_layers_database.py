"""
IHC Cell Layer Analysis (Config-driven, with versioned outputs & metadata)

- Loads IHC images from URLs stored in the HPA database.
- Uses tissue and cell masks from prediction pipeline outputs.
- Computes epidermal layers, cell diameters, DAB/H intensities.
- Saves processed results into SQLite database defined in config.
- Saves figures and metadata to versioned output folder.

CLI:
python cell_layers_database.py --version v001 --gene TP63
"""

import os
import json
import sqlite3
import argparse
import urllib.request
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, label
from skimage import measure, img_as_ubyte
from skimage.color import rgb2hed, hed2rgb
import cv2

from .. import config_helper

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_image(url):
    """Download image from URL as RGB."""
    try:
        with urllib.request.urlopen(url) as r:
            arr = np.asarray(bytearray(r.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"[ERROR] Failed to load {url}: {e}")
        return None


def load_image_and_masks(url, mask_path, cell_mask_path):
    """Download image and load tissue/cell masks."""
    rgb_img = load_image(url)
    if rgb_img is None:
        return None, None, None

    if not os.path.exists(mask_path) or not os.path.exists(cell_mask_path):
        print(f"[ERROR] Missing mask file: {mask_path} or {cell_mask_path}")
        return None, None, None

    tissue_mask = np.load(mask_path)["mask"]
    tissue_mask = correct_mask(tissue_mask)

    cell_mask = np.load(cell_mask_path)["mask"]

    return rgb_img, tissue_mask, cell_mask


def correct_class(mask, class_value):
    class_mask = (mask == class_value)
    labeled_mask, _ = label(class_mask)
    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0
    threshold = sizes.max() / 8
    new_mask = np.zeros_like(mask)
    for lbl, size in enumerate(sizes):
        if size >= threshold:
            new_mask[labeled_mask == lbl] = class_value
    return new_mask


def correct_mask(mask):
    mask1 = correct_class(mask, 1)
    mask2 = correct_class(mask, 2)
    new_mask = np.zeros_like(mask)
    new_mask += mask1
    new_mask += mask2
    return new_mask.astype(mask.dtype)


def compute_distance_transform(mask):
    epidermis_mask = (mask == 2)
    dist = distance_transform_edt(~(mask == 1))
    dist[~epidermis_mask] = 0
    return dist, epidermis_mask


def calculate_cell_diameter(cell_mask):
    labels = measure.label(cell_mask)
    props = measure.regionprops(labels)
    diameters = [p.equivalent_diameter for p in props]
    mean_diameter = np.mean(diameters)
    print(f"[INFO] Mean cell diameter: {mean_diameter:.2f}")
    return mean_diameter


def divide_into_layers(distance, epidermis_mask, mean_cell_diameter):
    num_layers = int(np.ceil(distance.max() / mean_cell_diameter))
    layered = np.zeros_like(distance)
    intervals = np.linspace(0, distance.max(), num_layers + 1)
    for i in range(num_layers):
        mask_layer = (distance >= intervals[i]) & (distance < intervals[i + 1])
        layered[mask_layer & epidermis_mask] = i + 1
    print(f"[INFO] Number of layers: {num_layers}")
    return layered, num_layers


def color_separate(rgb_img):
    hed = rgb2hed(rgb_img)
    null = np.zeros_like(hed[:, :, 0])
    H = img_as_ubyte(hed2rgb(np.stack((hed[:, :, 0], null, null), axis=-1)))
    D = img_as_ubyte(hed2rgb(np.stack((null, null, hed[:, :, 2]), axis=-1)))
    H = np.invert(H[:, :, 2])
    D = np.invert(D[:, :, 2])
    return H, D


def calculate_intensity_per_layer(layered_mask, channel, num_layers):
    intensities = []
    for i in range(1, num_layers + 1):
        mask = (layered_mask == i)
        intensities.append(np.mean(channel[mask]))
    return intensities


def save_to_database(db_path, main_id, mean_cell_diameter, num_layers, d_intensities, h_intensities, ratio_intensities):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_data (
            mainID TEXT PRIMARY KEY,
            mean_cell_diameter REAL,
            num_layers INTEGER,
            d_intensities TEXT,
            h_intensities TEXT,
            ratio_intensities TEXT
        )
    ''')
    cursor.execute("SELECT COUNT(*) FROM processed_data WHERE mainID=?", (main_id,))
    d_json = json.dumps(d_intensities)
    h_json = json.dumps(h_intensities)
    ratio_json = json.dumps(ratio_intensities.tolist())
    if cursor.fetchone()[0]:
        cursor.execute('''
            UPDATE processed_data SET mean_cell_diameter=?, num_layers=?, d_intensities=?, h_intensities=?, ratio_intensities=?
            WHERE mainID=?
        ''', (mean_cell_diameter, num_layers, d_json, h_json, ratio_json, main_id))
        print(f"[INFO] Updated record: {main_id}")
    else:
        cursor.execute('''
            INSERT INTO processed_data (mainID, mean_cell_diameter, num_layers, d_intensities, h_intensities, ratio_intensities)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (main_id, mean_cell_diameter, num_layers, d_json, h_json, ratio_json))
        print(f"[INFO] Inserted record: {main_id}")
    conn.commit()
    conn.close()

def save_figure(output_dir, filename,
                rgb_img, layered_mask,
                H, D,
                d_intensities, h_intensities,
                gene_name, main_id,
                mean_diameter, num_layers):

    ensure_dir(output_dir)

    ratio = np.array(d_intensities) / (np.array(h_intensities) + 1e-6)

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(
        3, 4,
        width_ratios=[1, 1, 1, 1.3],
        hspace=0.5,
        wspace=0.35
    )

    # ------------------------------------------------------------------
    # Global title (never clipped)
    # ------------------------------------------------------------------
    fig.suptitle(
        f"{gene_name} | mainID: {main_id}\n"
        f"Mean cell diameter: {mean_diameter:.2f}px | Layers: {num_layers}",
        fontsize=20,
        y=0.97
    )

    # ======================= Row 1 =======================
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb_img)
    ax.set_title("Original IHC")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(rgb_img)
    ax.imshow(layered_mask, cmap="jet", alpha=0.5)
    ax.set_title("Layered Epidermis")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(D, cmap="gray")
    ax.set_title("DAB Channel")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 3])
    ax.plot(range(1, num_layers + 1), d_intensities, marker="o")
    ax.set_title("Mean DAB Intensity per Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Intensity")
    ax.set_xticks(range(1, num_layers + 1))
    ax.grid(True)

    # ======================= Row 2 =======================
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(H, cmap="gray")
    ax.set_title("H Channel")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 3])
    ax.plot(range(1, num_layers + 1), h_intensities, marker="o", color="orange")
    ax.set_title("Mean H Intensity per Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Intensity")
    ax.set_xticks(range(1, num_layers + 1))
    ax.grid(True)

    # ======================= Row 3 =======================
    ax = fig.add_subplot(gs[2, 3])
    ax.plot(range(1, num_layers + 1), ratio, marker="s", color="crimson")

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(
        0.5 * ratio.min(),
        1.5 * ratio.max()
    )

    ax.set_title("D / H Intensity Ratio per Layer (1 = Balanced)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("D / H Ratio")
    ax.set_xticks(range(1, num_layers + 1))
    ax.grid(True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    plt.savefig(
        os.path.join(output_dir, f"{filename}.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)


# -------------------------------------------------------------------------
# Image Processing
# -------------------------------------------------------------------------
def process_image_pipeline(config, filename, main_id, gene_name, url, output_dir):
    mask_path = os.path.join(config['paths']['masks_ihc'], f"{filename}.npz")
    cell_mask_path = os.path.join(config['paths']['masks_cells'], f"{filename}.npz")
    rgb_img, mask, cell_mask = load_image_and_masks(url, mask_path, cell_mask_path)
    if rgb_img is None:
        return filename

    mean_diameter = calculate_cell_diameter(cell_mask)
    dist, epidermis_mask = compute_distance_transform(mask)
    layered_mask, num_layers = divide_into_layers(dist, epidermis_mask, mean_diameter)
    H, D = color_separate(rgb_img)
    d_intensities = calculate_intensity_per_layer(layered_mask, D, num_layers)
    h_intensities = calculate_intensity_per_layer(layered_mask, H, num_layers)
    ratio_intensities = np.array(d_intensities) / np.array(h_intensities)

    save_to_database(config['paths']['database_path'], main_id, mean_diameter, num_layers,
                     d_intensities, h_intensities, ratio_intensities)
    save_figure(
        output_dir, filename,
        rgb_img, layered_mask,
        H, D,
        d_intensities, h_intensities,
        gene_name=gene_name,
        main_id=main_id,
        mean_diameter=mean_diameter,
        num_layers=num_layers
    )

    return None


# -------------------------------------------------------------------------
# CLI / Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_cfg = os.path.join(PROJECT_ROOT, "config", "base_config.yaml")
    epi_cfg = os.path.join(PROJECT_ROOT, "config", "epidermis_analysis_config.yaml")

    config = config_helper.ConfigLoader.load_config(base_cfg, epi_cfg)

    parser = argparse.ArgumentParser(description="IHC Cell Layer Analysis")
    parser.add_argument("--version", required=True, help="Dataset / model version, e.g. v001")
    parser.add_argument("--gene", default=None, help="Optional gene filter (e.g. TP63)")
    args = parser.parse_args()

    output_root = ensure_dir(os.path.join(config['paths']['data_root'], args.version, "layer_analysis"))

    conn = sqlite3.connect(config['paths']['database_path'])
    cursor = conn.cursor()
    if args.gene:
        cursor.execute(
            "SELECT mainID, name, url FROM elGenes WHERE name=? ORDER BY mainID ASC",
            (args.gene,)
        )
    else:
        cursor.execute(
            "SELECT mainID, name, url FROM elGenes ORDER BY mainID ASC"
        )
    data = cursor.fetchall()
    conn.close()

    failed_files = []
    for main_id, name, url in data:
        filename = url.split('/')[-2] + '-' + url.split('/')[-1].split('.')[0]
        try:
            failed = process_image_pipeline(
                config=config,
                filename=filename,
                main_id=main_id,
                gene_name=name,
                url=url,
                output_dir=output_root
            )
            if failed:
                failed_files.append((main_id, failed))
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            failed_files.append((main_id, filename))

    metadata_path = os.path.join(output_root, "run_metadata.json")
    metadata = {
        "version": args.version,
        "gene": args.gene,
        "processed_at": datetime.now().isoformat(),
        "processed_count": len(data) - len(failed_files),
        "failed_count": len(failed_files),
        "failed_files": failed_files
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if failed_files:
        print(f"[INFO] {len(failed_files)} files failed. See run_metadata.json for details.")
