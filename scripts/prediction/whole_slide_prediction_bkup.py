"""
Whole-Slide Image Prediction Pipeline
-------------------------------------

Performs prediction on whole-slide images using a pre-trained segmentation model.

Features:
- Loads images from URLs stored in a SQLite database.
- Predicts segmentation masks for multiple classes.
- Post-processes masks to keep only the largest connected regions and
  correct overlapping epidermis/keratin classes.
- Crops and re-predicts the largest regions for refined masks.
- Saves corrected masks as compressed `.npz`.
- Generates multi-panel visualizations for each image.
- Resumes processing from the last processed database ID.
- Supports versioned models.
"""

import os
import urllib.request
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

from skimage.morphology import label as sk_label
from tensorflow.keras.models import load_model
import segmentation_models as sm
#from tkinter import Tk, filedialog
from datetime import datetime
from pathlib import Path


from .. import config_helper

# ----------------------
# Helper functions
# ----------------------
def create_connection(db_file):
    try:
        return sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def correct_mask(mask):
    """Combine epidermis (2) and keratin (3) regions, preserve mixed areas."""
    combined_mask = (mask == 2) | (mask == 3)
    labeled_array, num_features = sk_label(combined_mask, return_num=True)

    for i in range(1, num_features + 1):
        region = (labeled_array == i)
        if np.any(mask[region] == 2) and np.any(mask[region] == 3):
            continue
        mask[region] = 1 if np.any(mask[region] == 2) else 0
    return mask

def keep_largest_combined_area(pred_mask):
    """Keep only the largest connected component in the predicted mask."""
    binary_mask = (pred_mask != 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(pred_mask)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    cv2.drawContours(largest_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
    return largest_mask

def crop_and_predict(original_image, predicted_mask, target_size, model, backbone):
    """Crop largest predicted region and re-run prediction for refinement."""
    pred_uint8 = (predicted_mask != 0).astype(np.uint8)
    contours, _ = cv2.findContours(pred_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return original_image, predicted_mask

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    scale_x = original_image.shape[1] / predicted_mask.shape[1]
    scale_y = original_image.shape[0] / predicted_mask.shape[0]

    x_o, y_o, w_o, h_o = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
    cropped_image = original_image[y_o:y_o + h_o, x_o:x_o + w_o]
    resized_cropped = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_NEAREST)

    input_img = np.expand_dims(resized_cropped, 0)
    input_img = sm.get_preprocessing(backbone)(input_img)

    pred_cropped = model.predict(input_img)
    pred_cropped = np.argmax(pred_cropped, axis=3)[0, :, :]
    resized_pred = cv2.resize(pred_cropped, (cropped_image.shape[1], cropped_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    combined_mask = np.zeros_like(original_image[:, :, 0], dtype=np.uint8)
    combined_mask[y_o:y_o + resized_pred.shape[0], x_o:x_o + resized_pred.shape[1]] = resized_pred

    return cropped_image, combined_mask

def save_last_processed_id(last_id, filename):
    with open(filename, 'w') as f:
        f.write(str(last_id))

def load_last_processed_id(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return int(f.read())
    return 0

def save_visualization(original, masks, gene_name, mainID, output_dir):
    """Save multi-panel figure: original, predicted, largest, cropped, corrected masks."""
    plt.figure(figsize=(18, 4))
    plt.suptitle(f"{gene_name} | ID: {mainID}", fontsize=16)
    titles = ["Original Image", "Predicted Mask", "Largest Area Mask",
              "Cropped Image", "Cropped Mask", "Corrected Mask"]

    for i, mask in enumerate(masks):
        plt.subplot(1, 6, i + 1)
        plt.title(titles[i])
        if i == 0 or i == 3:
            # show original or cropped image normally
            plt.imshow(mask)
        else:
            # class mask: no colormap → restore original behavior
            plt.imshow(mask, vmin=0, vmax=3)
        plt.axis('off')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{gene_name}_{mainID}.png"))
    plt.close()


def process_entry(mainID, gene_name, image_source, model, backbone,
                  target_size, output_vis_dir, output_masks_dir):
    # Load image (file path OR URL)
    if image_source.startswith("http"):
        resp = urllib.request.urlopen(image_source)
        arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        ihc_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        ihc_img = cv2.imread(image_source)

    ihc_img = cv2.cvtColor(ihc_img, cv2.COLOR_BGR2RGB)

    # Predict base mask
    ihc_resized = cv2.resize(ihc_img, target_size, interpolation=cv2.INTER_NEAREST)

    ihc_resized = ihc_resized.astype(np.float32)

    input_img = np.expand_dims(ihc_resized, 0)
    input_img = sm.get_preprocessing(backbone)(input_img)

    pred_mask = model.predict(input_img)
    pred_mask = np.argmax(pred_mask, axis=3)[0, :, :]

    # Post-processing
    largest_mask = keep_largest_combined_area(pred_mask)
    cropped_img, combined_mask = crop_and_predict(
        ihc_img, largest_mask, target_size, model, backbone
    )
    corrected_mask = correct_mask(combined_mask)

    # Save mask
    mask_file = os.path.join(output_masks_dir, f"{gene_name}_{mainID}.npz")
    np.savez_compressed(mask_file, mask=corrected_mask)

    # Save visualization
    save_visualization(
        ihc_img,
        [ihc_img, pred_mask, largest_mask, cropped_img, combined_mask, corrected_mask],
        gene_name, mainID, output_vis_dir
    )

    print(f"Processed {gene_name} | ID: {mainID}")


def load_from_folder():
    folder = input("📁 Enter the path to the folder containing test images: ").strip()
    if not folder or not os.path.isdir(folder):
        raise ValueError(f"❌ Folder not found: {folder}")


    valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    files = [f for f in os.listdir(folder) if f.lower().endswith(valid_ext)]

    if not files:
        raise ValueError("❌ No valid image files in folder.")

    results = []
    for idx, filename in enumerate(sorted(files), start=1):
        stem = Path(filename).stem
        full_path = os.path.join(folder, filename)
        results.append((idx, stem, full_path))  # fake mainID, gene_name, path

    print(f"Found {len(results)} images.")

    return results
def load_from_database(config, last_id):
    conn = create_connection(config['paths']['database_path'])
    cur = conn.cursor()

    cur.execute(
        "SELECT mainID, name, url FROM elGenes WHERE mainID > ?",
        (last_id,)
    )
    rows = cur.fetchall()

    conn.close()
    return rows

# ----------------------
# Main pipeline
# ----------------------
def main(config, version="v001"):
    target_size = (512, 512)
    backbone = config['training_params']['model_type']

    # --- Model picker ---
    model_dir = os.path.join(config['paths']['models'], f"{backbone}_{version}")
    print(f"📂 Model directory: {model_dir}")
    model_file = input("Enter the full path to the .keras model file: ").strip()

    if not model_file or not os.path.isfile(model_file):
        raise ValueError(f"❌ Model file not found: {model_file}")

    model = load_model(model_file, compile=False)
    print(f"✅ Loaded model: {model_file}")

    # Output directories
    model_name = Path(model_file).stem
    base_pred_dir = os.path.join(config['paths']['data_root'], version, "predictions", model_name)
    output_vis_dir = os.path.join(base_pred_dir, "outputs_visualization")
    output_masks_dir = os.path.join(base_pred_dir, "outputs_masks")
    os.makedirs(output_vis_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    # Save metadata

    with open(os.path.join(base_pred_dir, "metadata.json"), "w") as f:
        json.dump({
            "model_file": model_file,
            "model_name": model_name,
            "version": version,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    # ------------------------------
    # ASK USER FOR MODE
    # ------------------------------
    print("\nSelect mode:\n1 = Database mode\n2 = Folder test/debug mode\n")
    mode = input("Enter mode (1/2): ").strip()

    last_id_file = os.path.join(base_pred_dir, "last_processed_id.txt")
    last_id = load_last_processed_id(last_id_file)

    if mode == "1":
        entries = load_from_database(config, last_id)
        convert = lambda row: (row[0], row[1], row[2])  # no change
    else:
        entries = load_from_folder()
        convert = lambda row: (row[0], row[1], row[2])  # already (id, name, path)

    # ------------------------------
    # PROCESS LOOP
    # ------------------------------
    for mainID, gene_name, src in entries:
        try:
            process_entry(
                mainID, gene_name, src,
                model, backbone,
                target_size,
                output_vis_dir, output_masks_dir
            )
            save_last_processed_id(mainID, last_id_file)

        except Exception as e:
            print(f"❌ Error processing {gene_name} (ID {mainID}): {e}")

    print("Prediction completed.")


# ----------------------
# Entry point
# ----------------------
if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_path = os.path.join(PROJECT_ROOT, "config", "base_config.yaml")
    config_path = os.path.join(PROJECT_ROOT, "config", "prediction_config.yaml")

    config = config_helper.ConfigLoader.load_config(base_path, config_path)

    import argparse
    parser = argparse.ArgumentParser(description="Whole-slide prediction")
    parser.add_argument("--version", type=str, default="v001", help="Model version to use")
    args = parser.parse_args()

    main(config, version=args.version)
