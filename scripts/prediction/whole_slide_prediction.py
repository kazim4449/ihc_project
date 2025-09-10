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

from skimage.morphology import label as sk_label
from tensorflow.keras.models import load_model
import segmentation_models as sm

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
        plt.imshow(mask if i == 0 or i == 3 else mask, cmap='nipy_spectral')
        plt.axis('off')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{gene_name}_{mainID}.png"))
    plt.close()

# ----------------------
# Main pipeline
# ----------------------
def main(config, version="v001"):
    target_size = (512, 512)
    backbone = config['training_params']['model_type']

    # Load model matching version
    model_dir = os.path.join(config['paths']['models'], f"{backbone}_{version}")
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
    if len(model_files) != 1:
        raise ValueError(f"Expected one .keras model in {model_dir}, found {len(model_files)}")
    model_file = os.path.join(model_dir, model_files[0])
    model = load_model(model_file, compile=False)
    print(f"Loaded model: {model_file}")

    # Connect to database
    conn = create_connection(config['paths']['database_path'])
    cur = conn.cursor()

    # Versioned folders for outputs and last_processed_id
    output_vis_dir = os.path.join(config['paths']['prediction_data'], version, "outputs_visualization")
    output_masks_dir = os.path.join(config['paths']['prediction_data'], version, "outputs_masks")
    os.makedirs(output_vis_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    last_id_file = os.path.join(config['paths']['prediction_data'], version, "last_processed_id.txt")
    last_id = load_last_processed_id(last_id_file)

    # Fetch rows to process
    cur.execute("SELECT mainID, name, url FROM elGenes WHERE mainID > ?", (last_id,))
    rows = cur.fetchall()

    for mainID, gene_name, url in rows:
        try:
            # Load image from URL
            resp = urllib.request.urlopen(url)
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            ihc_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            ihc_img = cv2.cvtColor(ihc_img, cv2.COLOR_BGR2RGB)

            # Resize & preprocess
            ihc_resized = cv2.resize(ihc_img, target_size, interpolation=cv2.INTER_NEAREST)
            input_img = np.expand_dims(ihc_resized, 0)
            input_img = sm.get_preprocessing(backbone)(input_img)

            # Predict mask
            pred_mask = model.predict(input_img)
            pred_mask = np.argmax(pred_mask, axis=3)[0, :, :]

            # Post-processing
            largest_mask = keep_largest_combined_area(pred_mask)
            cropped_img, combined_mask = crop_and_predict(ihc_img, largest_mask, target_size, model, backbone)
            corrected_mask = correct_mask(combined_mask)

            # Save mask
            mask_filename = f"{gene_name}_{mainID}.npz"
            np.savez_compressed(os.path.join(output_masks_dir, mask_filename), mask=corrected_mask)

            # Save visualization
            save_visualization(ihc_img,
                               [ihc_img, pred_mask, largest_mask, cropped_img, combined_mask, corrected_mask],
                               gene_name, mainID, output_vis_dir)

            # Update last processed ID
            save_last_processed_id(mainID, last_id_file)
            print(f"Processed {gene_name} | ID: {mainID}")

        except Exception as e:
            print(f"Error processing {gene_name} | ID {mainID}: {e}")

    conn.close()
    print("Whole-slide prediction completed.")

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
