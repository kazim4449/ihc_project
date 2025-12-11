# ------------------------------
# Full Evaluation Metrics
# ------------------------------

import numpy as np  
import matplotlib.pyplot as plt
import os
import csv
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, matthews_corrcoef
)

def evaluate_metrics(model, X_val, y_val, eval_dir):
    preds = model.predict(X_val)
    if preds.ndim == 4:
        preds = np.argmax(preds, axis=-1)
    if y_val.ndim == 4:
        y_val_labels = np.argmax(y_val, axis=-1)
    else:
        y_val_labels = y_val

    y_true_flat = y_val_labels.flatten()
    y_pred_flat = preds.flatten()
    n_classes = len(np.unique(y_true_flat))

    # Confusion Matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=list(range(n_classes)))
    # Convert to percentage (row-wise normalization)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    # special annotation with raw counts of cm matrix on paranthesis along with cm_percent values 
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm_percent[i, j]:.3f}%\n({cm[i, j]})'
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=annot, fmt='', cmap="Blues",
                xticklabels=[f"Class {i}" for i in range(n_classes)],
                yticklabels=[f"Class {i}" for i in range(n_classes)])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    cm_file = os.path.join(eval_dir, "confusion_matrix.png")
    plt.savefig(cm_file); plt.close()
    print(f"Confusion matrix saved to: {cm_file}")

    # Classification Report
    report = classification_report(y_true_flat, y_pred_flat,
                                    target_names=[f"Class {i}" for i in range(n_classes)],
                                    digits=4)
    report_file = os.path.join(eval_dir, "classification_report.txt")
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Classification report saved to: {report_file}")

    # Precision, Recall, F1
    precision = precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)

    metrics_file = os.path.join(eval_dir, "metrics_summary.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
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
        for i in range(n_classes):
            TP = cm[i, i]
            FN = np.sum(cm[i, :]) - TP
            FP = np.sum(cm[:, i]) - TP
            TN = cm.sum() - (TP + FP + FN)

            # Compute other metrics
            precision_i = TP / (TP + FP + 1e-8)
            recall_i = TP / (TP + FN + 1e-8)
            f1_i = 2 * precision_i * recall_i / (precision_i + recall_i + 1e-8)
            IoU = TP / (TP + FP + FN + 1e-8)

            # MCC using sklearn (robust)
            y_true = [1] * TP + [1] * FN + [0] * FP + [0] * TN
            y_pred = [1] * TP + [0] * FN + [1] * FP + [0] * TN
            mcc_i = matthews_corrcoef(y_true, y_pred)

            precisions.append(precision_i)
            recalls.append(recall_i)
            f1s.append(f1_i)
            ious.append(IoU)
            mccs.append(mcc_i)

            writer.writerow([
                f"Class {i}", TP, FP, FN, TN,
                f"{precision_i:.4f}", f"{recall_i:.4f}", f"{f1_i:.4f}", f"{IoU:.4f}", f"{mcc_i:.4f}"
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