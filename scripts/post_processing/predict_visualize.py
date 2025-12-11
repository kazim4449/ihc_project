# ------------------------------
# Predictions & Visualization
# ------------------------------

import os
import numpy as np  
import matplotlib.pyplot as plt
from tensorflow import keras
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, matthews_corrcoef
)


def compute_per_class_iou(cm):
    """
    Computes per-class IoU and mean IoU from a confusion matrix.
    
    Parameters:
        cm (ndarray): Confusion matrix of shape (n_classes, n_classes)
    
    Returns:
        per_class_iou (list of floats)
        mean_iou (float)
    """
    per_class_iou = []

    for i in range(len(cm)):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        denom = TP + FP + FN
        iou = TP / denom if denom != 0 else 0.0
        per_class_iou.append(iou)

    return per_class_iou

def visualize_predictions(model, X_val, y_val, n_samples=3, save_path=None):
    n_samples = min(n_samples, X_val.shape[0])  # avoid out-of-bounds
    n_classes = y_val.shape[-1] if y_val.ndim == 4 else len(np.unique(y_val))
    preds = model.predict(X_val)
    preds_labels = np.argmax(preds, axis=-1)
    y_val_labels = np.argmax(y_val, axis=-1) if y_val.ndim == 4 else y_val
    # 3 images in a subplot 
    n0 = 3
    plt.figure(figsize=(12, n0 * 4))
    for i in range(n0):
        img = X_val[i].astype(np.uint8)
        true_mask = y_val_labels[i]
        pred_mask = preds_labels[i]

        # Display images
        plt.subplot(n_samples, 3, i*3+1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis('off')

        plt.subplot(n_samples, 3, i*3+2)
        plt.imshow(true_mask)
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(n_samples, 3, i*3+3)
        plt.imshow(pred_mask)
        plt.title("Prediction")
        plt.axis('off')

    plt.tight_layout()
    if save_path:
        pred_fig_path = os.path.join(save_path, "sample_predictions.png")
        plt.savefig(pred_fig_path)
        print(f"Sample predictions saved to {pred_fig_path}")
    plt.close()
    # for all samples 
    for i in range(n_samples):

        # Plotting the images
        img = X_val[i].astype(np.uint8)
        true_mask = y_val_labels[i]
        pred_mask = preds_labels[i]

        # Flatten masks for metrics
        true_flat = true_mask.flatten()
        pred_flat = pred_mask.flatten()

        # Per-class metrics
        cm = confusion_matrix(true_flat, pred_flat, labels=list(range(n_classes)))
        precision = precision_score(true_flat, pred_flat, average='weighted', zero_division=0)
        recall = recall_score(true_flat, pred_flat, average='weighted', zero_division=0)
        f1 = f1_score(true_flat, pred_flat, average='weighted', zero_division=0)

#############################################################################################################################
        # this is not correct 
        # Per-class IOU using keras MeanIoU
        IOU_keras = keras.metrics.MeanIoU(num_classes=n_classes)
        IOU_keras.update_state(true_mask, pred_mask)
        iou_matrix = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
        per_class_iou = [iou_matrix[j,j] / (np.sum(iou_matrix[j,:]) + 1e-8) for j in range(n_classes)]

#############################################################################################################################

        per_class_iou_new = compute_per_class_iou(cm)
#############################################################################################################################


        # Save metrics and confusion matrices for each sample
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            metrics_file = os.path.join(save_path, f"sample_{i}_metrics.txt")
            with open(metrics_file, "w") as f:
                f.write(f"Weighted Precision: {precision:.4f}\n")
                f.write(f"Weighted Recall: {recall:.4f}\n")
                f.write(f"Weighted F1: {f1:.4f}\n")
                f.write(f"Per-class IOU:\n")
                for c, iou_val in enumerate(per_class_iou):
                    f.write(f"Class {c}: {iou_val:.4f}\n")
                f.write(f"Per-class IOU_using_cm:\n")
                for c, iou_val in enumerate(per_class_iou_new):
                    f.write(f"Class_cm_{c}: {iou_val:.4f}\n")

            cm_file = os.path.join(save_path, f"sample_{i}_confusion_matrix.png")

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=[f"Class {j}" for j in range(n_classes)],
                        yticklabels=[f"Class {j}" for j in range(n_classes)])
            plt.title(f"Sample {i} Confusion Matrix")
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.savefig(cm_file)
            plt.close()

            plt.figure(figsize=(12,4))
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            # Display images
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title("Image")
            plt.axis('off')

            plt.subplot(1, 3,2)
            plt.imshow(true_mask)
            plt.title("Ground Truth")
            plt.axis('off')

            plt.subplot(1, 3,3)
            plt.imshow(pred_mask)
            plt.title("Prediction")
            plt.axis('off')
            plt.tight_layout()
            sample_image_file = os.path.join(save_path, f"sample_{i}.png")       
            plt.savefig(sample_image_file)
            print(f"Sample predictions saved to {sample_image_file}")
            plt.close()
            
        


