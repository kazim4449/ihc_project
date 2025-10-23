import numpy as np
import os
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score,matthews_corrcoef
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def visualize_predictions(model, X_val, y_val, n_samples=3, save_path=None):
    """
    Visualize model predictions for n_samples and save metrics.
    Mirrors old test.py behavior.
    """

    n_samples = min(n_samples, X_val.shape[0])
    if y_val.ndim == 4:
        y_true_flat = np.argmax(y_val, axis=-1)
    else:
        y_true_flat = y_val

    # Determine number of classes
    labels = np.unique(y_true_flat)
    n_classes = len(labels)

    preds = model.predict(X_val)
    if preds.ndim == 4:
        preds_labels = np.argmax(preds, axis=-1)
    else:
        preds_labels = preds

    os.makedirs(save_path, exist_ok=True) if save_path else None

    for i in range(n_samples):
        img = X_val[i].astype(np.uint8)
        true_mask = y_true_flat[i]
        pred_mask = preds_labels[i]

        # -------------------------------
        # Label encoding to ensure 0..n_classes-1
        # -------------------------------
        true_mask_enc = true_mask  # 1..4
        pred_mask_enc = pred_mask + 1  # shift 0..3 -> 1..4

        # -------------------------------
        # Mean IoU
        # -------------------------------
        IOU_keras = tf.keras.metrics.MeanIoU(num_classes=n_classes)
        IOU_keras.update_state(true_mask_enc, pred_mask_enc)
        iou_matrix = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
        per_class_iou = [iou_matrix[j, j] / (np.sum(iou_matrix[j, :]) + 1e-8) for j in range(n_classes)]
        mean_iou = IOU_keras.result().numpy()

        # -------------------------------
        # Metrics: precision, recall, f1
        # -------------------------------
        precision = precision_score(true_mask_enc.flatten(), pred_mask_enc.flatten(), average='weighted', zero_division=0)
        recall = recall_score(true_mask_enc.flatten(), pred_mask_enc.flatten(), average='weighted', zero_division=0)
        f1 = f1_score(true_mask_enc.flatten(), pred_mask_enc.flatten(), average='weighted', zero_division=0)

        # -------------------------------
        # Save metrics
        # -------------------------------
        if save_path:
            metrics_file = os.path.join(save_path, f"{i}_metrics.txt")
            with open(metrics_file, "w") as f:
                f.write(f"Weighted Precision: {precision:.4f}\n")
                f.write(f"Weighted Recall: {recall:.4f}\n")
                f.write(f"Weighted F1: {f1:.4f}\n")
                f.write(f"Mean IoU: {mean_iou:.4f}\n")
                f.write("Per-class IoU:\n")
                for cls_idx, iou_val in enumerate(per_class_iou):
                    f.write(f"Class {cls_idx + 1}: {iou_val:.4f}\n")

        # -------------------------------
        # Confusion Matrix
        # -------------------------------
        cm = confusion_matrix(true_mask_enc.flatten(), pred_mask_enc.flatten(), labels=list(range(n_classes)))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[f"Class {j+1}" for j in range(n_classes)],
                    yticklabels=[f"Class {j+1}" for j in range(n_classes)])
        plt.title(f"Sample {i} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        if save_path:
            cm_file = os.path.join(save_path, f"{i}_confusion_matrix.png")
            plt.savefig(cm_file)
        plt.close()

        # -------------------------------
        # Visualization
        # -------------------------------
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(true_mask_enc)
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask_enc)
        plt.title("Prediction")
        plt.axis("off")

        if save_path:
            pred_file = os.path.join(save_path, f"{i}_prediction.png")
            plt.savefig(pred_file)
        plt.close()
