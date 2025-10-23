import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import segmentation_models as sm
from tkinter import Tk, filedialog

from .. import config_helper
from .. import utils



def create_training_labels(mask):
    """Map RGB mask colors into 4 classes: BG=1, Pap=2, Epi=3, Ker=4"""
    bg_color = [0, 0, 0]
    epi_color = [73, 0, 106]
    ker_color = [248, 123, 168]
    pap_color = [236, 85, 157]

    bcc_color = [127, 255, 255]
    inf_color = [145, 1, 122]
    gld_color = [108, 0, 115]
    iec_color = [255, 127, 127]
    ret_color = [181, 9, 130]

    fol_color = [216, 47, 148]
    hpy_color = [254, 246, 242]
    scc_color = [127, 255, 142]

    labels = np.full(mask.shape[:2], 1, dtype=np.uint8)  # default BG

    labels[np.all(mask == pap_color, axis=-1)] = 2
    labels[np.all(mask == epi_color, axis=-1)] = 3
    labels[np.all(mask == ker_color, axis=-1)] = 4

    labels[np.all(mask == iec_color, axis=-1)] = 3
    labels[np.all(mask == bcc_color, axis=-1)] = 3
    labels[np.all(mask == inf_color, axis=-1)] = 2
    labels[np.all(mask == gld_color, axis=-1)] = 2
    labels[np.all(mask == ret_color, axis=-1)] = 2

    labels[np.all(mask == fol_color, axis=-1)] = 3
    labels[np.all(mask == hpy_color, axis=-1)] = 2
    labels[np.all(mask == scc_color, axis=-1)] = 2

    print(np.unique(labels), 'CLASSES 1-4')
    return labels

def log_test_results(model, X_test, y_test, save_path=None, n_samples=None, labels=None, target_names=None):
    import csv
    from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, matthews_corrcoef
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    if save_path is None:
        save_path = os.getcwd()
    os.makedirs(save_path, exist_ok=True)

    # Create subfolders
    pred_dir = os.path.join(save_path, "test_predictions")
    eval_dir = os.path.join(save_path, "test_evaluation")
    for d in [pred_dir, eval_dir]:
        os.makedirs(d, exist_ok=True)

    # ------------------------------
    # Predictions
    # ------------------------------
    preds = model.predict(X_test)
    if preds.ndim == 4:
        preds = np.argmax(preds, axis=-1)
    if y_test.ndim == 4:
        y_test_labels = np.argmax(y_test, axis=-1)
    else:
        y_test_labels = y_test

    y_true_flat = y_test_labels.flatten()
    y_pred_flat = preds.flatten()

    # Determine labels and target_names
    if labels is None:
        labels = np.unique(y_true_flat)
    if target_names is None:
        target_names = [f"Class {i}" for i in labels]
    n_classes = len(labels)

    # ------------------------------
    # Confusion Matrix
    # ------------------------------
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    cm_file = os.path.join(eval_dir, "confusion_matrix.png")
    plt.savefig(cm_file); plt.close()
    print(f"Confusion matrix saved to: {cm_file}")

    # ------------------------------
    # Classification Report
    # ------------------------------
    report = classification_report(y_true_flat, y_pred_flat,
                                   labels=labels,
                                   target_names=target_names)
    report_file = os.path.join(eval_dir, "classification_report.txt")
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Classification report saved to: {report_file}")

    # ------------------------------
    # Weighted metrics
    # ------------------------------
    precision = precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    metrics_file = os.path.join(eval_dir, "metrics_summary.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Weighted Precision: {precision:.4f}\n")
        f.write(f"Weighted Recall: {recall:.4f}\n")
        f.write(f"Weighted F1: {f1:.4f}\n")
    print(f"Metrics summary saved to: {metrics_file}")

    # ------------------------------
    # Per-class + Mean Metrics (CSV)
    # ------------------------------
    metrics_csv = os.path.join(eval_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Class", "TP", "FP", "FN", "TN",
            "Precision", "Recall", "F1", "IoU", "MCC"
        ])

        precisions, recalls, f1s, ious, mccs = [], [], [], [], []
        for idx, label in enumerate(labels):
            TP = cm[idx, idx]
            FN = np.sum(cm[idx, :]) - TP
            FP = np.sum(cm[:, idx]) - TP
            TN = cm.sum() - (TP + FP + FN)

            # Metrics
            precision_i = TP / (TP + FP + 1e-8)
            recall_i = TP / (TP + FN + 1e-8)
            f1_i = 2 * precision_i * recall_i / (precision_i + recall_i + 1e-8)
            IoU = TP / (TP + FP + FN + 1e-8)

            # MCC using sklearn
            y_true_bin = [1] * TP + [1] * FN + [0] * FP + [0] * TN
            y_pred_bin = [1] * TP + [0] * FN + [1] * FP + [0] * TN
            mcc_i = matthews_corrcoef(y_true_bin, y_pred_bin)

            precisions.append(precision_i)
            recalls.append(recall_i)
            f1s.append(f1_i)
            ious.append(IoU)
            mccs.append(mcc_i)

            writer.writerow([
                f"{target_names[idx]}", TP, FP, FN, TN,
                f"{precision_i:.4f}", f"{recall_i:.4f}",
                f"{f1_i:.4f}", f"{IoU:.4f}", f"{mcc_i:.4f}"
            ])

        # Mean row
        writer.writerow([
            "Mean", "-", "-", "-", "-",
            f"{np.mean(precisions):.4f}",
            f"{np.mean(recalls):.4f}",
            f"{np.mean(f1s):.4f}",
            f"{np.mean(ious):.4f}",
            f"{np.mean(mccs):.4f}"
        ])

    print(f"Unified metrics (per-class + mean) saved to: {metrics_csv}")

    # ------------------------------
    # Sample Predictions Visualization
    # ------------------------------
    if n_samples is None:
        n_samples = X_test.shape[0]
    utils.visualize_predictions(model, X_test, y_test, n_samples=n_samples, save_path=pred_dir)


class TestPipeline:
    def __init__(self, config, version="v001"):
        self.config = config
        self.version = version

        self.test_images_path = os.path.join(config['paths']['training_data'], version, "images", "test")
        self.test_masks_path = os.path.join(config['paths']['training_data'], version, "masks", "test")

        self.model_path = os.path.join(config['paths']['models'], f"{config['training_params']['model_type']}_{version}")
        self.n_classes = 4  # Hardcoded for BG, Pap, Epi, Ker
        self.img_size = config['training_params']['img_size']
        self.model_type = config['training_params']['model_type']

    def load_images_and_masks(self, images_dir, masks_dir):
        import cv2
        image_files = sorted(os.listdir(images_dir))
        mask_files = sorted(os.listdir(masks_dir))
        images, masks = [], []

        for img_file, mask_file in zip(image_files, mask_files):
            img = cv2.imread(os.path.join(images_dir, img_file), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

            mask = cv2.imread(os.path.join(masks_dir, mask_file), cv2.IMREAD_COLOR)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = create_training_labels(mask)

            images.append(img)
            masks.append(mask)
        return np.array(images), np.array(masks)

    def preprocess_data(self, images, masks):
        X_images = sm.get_preprocessing(self.model_type)(images)
        return X_images, masks.astype(np.int32)

    def run_test(self, n_samples=None):
        print("Starting test pipeline...")
        # Open file dialog
        root = Tk()
        root.withdraw()
        print("Opening file dialog to select .keras model...")
        model_file = filedialog.askopenfilename(
            initialdir=self.model_path,
            title="Select a .keras model file",
            filetypes=[("Keras models", "*.keras")]
        )
        root.destroy()

        if not model_file:
            raise FileNotFoundError("No model file selected!")
        print(f"Selected model: {model_file}")

        print("Loading model... (this may take a while for large models)")
        model = tf.keras.models.load_model(model_file, compile=False)
        print("Model loaded!")

        print("Loading test images and masks...")
        X_test, y_test = self.preprocess_data(
            *self.load_images_and_masks(self.test_images_path, self.test_masks_path))
        print(f"Loaded {X_test.shape[0]} test images and masks")

        test_save_path = os.path.join(self.model_path, "test",
                                      f"test_{os.path.splitext(os.path.basename(model_file))[0]}")
        os.makedirs(test_save_path, exist_ok=True)
        print(f"Test results will be saved to: {test_save_path}")

        print("Running test evaluation...")
        log_test_results(
            model,
            X_test,
            y_test,
            save_path=test_save_path,
            n_samples=n_samples,
            labels=[1, 2, 3, 4],  # Force 4 classes
            target_names=["BG", "Pap", "Epi", "Ker"]
        )
        print("Test evaluation completed!")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    parser = argparse.ArgumentParser(description="Evaluate trained segmentation model on test set")
    parser.add_argument("--version", type=str, default="v001", help="Version for evaluation outputs")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of test samples to visualize. Default is all samples")
    args = parser.parse_args()

    base_path = os.path.join(PROJECT_ROOT, "config", "base_config.yaml")
    train_path = os.path.join(PROJECT_ROOT, "config", "training_config.yaml")
    config = config_helper.ConfigLoader.load_config(base_path, train_path)

    tester = TestPipeline(config, version=args.version)
    tester.run_test(n_samples=args.n_samples)
