"""
Unified training script using base_config.yaml + training_config.yaml
Supports standard training, continuation, versioning, and Optuna hyperparameter search.
Includes training visualizations for metrics and sample predictions.
"""
import csv
import os
import json
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelEncoder
import segmentation_models as sm
from imgaug.augmenters import Sequential as iaaSequential, Flipud, Fliplr, Affine, GammaContrast, ChannelShuffle, GaussianBlur, ElasticTransformation
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import optuna
from optuna.integration import TFKerasPruningCallback
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score,matthews_corrcoef
import seaborn as sns

from .. import config_helper


# ----------------------
# Helper Classes
# ----------------------
class TrainingLabelCreator:
    """Creates training labels by mapping RGB mask values to class indices."""
    def __init__(self):
        self.color_mapping = {
            (0, 0, 0): 1,  # BG
            (236, 85, 157): 2,  # Pap
            (73, 0, 106): 3,  # Epi
            (248, 123, 168): 4,  # Ker
            (127, 255, 255): 3, (145, 1, 122): 2, (108, 0, 115): 2,
            (255, 127, 127): 3, (181, 9, 130): 2, (216, 47, 148): 3,
            (254, 246, 242): 2, (127, 255, 142): 2
        }

    def create_training_labels(self, mask):
        labels = np.full(mask.shape[:2], 1, dtype=np.uint8)  # default BG
        for color, label in self.color_mapping.items():
            labels[np.all(mask == np.array(color), axis=-1)] = label
        return labels


class CustomDataGenerator(tf.keras.utils.Sequence):
    """Custom generator for images/masks with optional augmentation."""
    def __init__(self, images, masks, batch_size, n_classes, augmentation=False):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.augmentation = augmentation

        # Define augmentation
        self.rot_flip_augmentation = [
            iaaSequential([]),
            Flipud(),
            Affine(rotate=90),
            Fliplr(),
            Affine(rotate=180),
            Flipud(),
            Affine(rotate=270),
            Flipud()
        ]
        self.color_augmentation = iaaSequential([
            GammaContrast(gamma=(0.5, 2.0), per_channel=True),
            ChannelShuffle(1.0),
            GaussianBlur(sigma=(0.0, 2.0)),
            ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
        ])

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        start, end = index * self.batch_size, (index + 1) * self.batch_size
        batch_images, batch_masks = self.images[start:end], self.masks[start:end]

        if self.augmentation:
            aug_imgs, aug_masks = [], []
            for i in range(len(batch_images)):
                imgs, masks = self.perform_augmentation(batch_images[i], batch_masks[i])
                aug_imgs.extend(imgs)
                aug_masks.extend(masks)
            batch_images = np.array(aug_imgs)
            batch_masks = np.array(aug_masks)

        batch_masks = to_categorical(batch_masks, num_classes=self.n_classes)
        return batch_images, batch_masks

    def perform_augmentation(self, image, mask):
        segmap = SegmentationMapsOnImage(mask, shape=image.shape)
        aug_images, aug_masks = [], []

        for aug in self.rot_flip_augmentation:
            color_img = self.color_augmentation(image=image)
            augmented = aug(image=color_img, segmentation_maps=segmap)
            aug_images.append(augmented[0])
            aug_masks.append(augmented[1].arr)
        return aug_images, aug_masks



# ----------------------
# Visualization helpers
# ----------------------
def log_training_results(model, history, X_val, y_val, save_path=None):
    if save_path is None:
        save_path = os.getcwd()
    os.makedirs(save_path, exist_ok=True)

    # Create subfolders
    training_dir = os.path.join(save_path, "training_curves")
    pred_dir = os.path.join(save_path, "sample_predictions")
    eval_dir = os.path.join(save_path, "evaluation")
    for d in [training_dir, pred_dir, eval_dir]:
        os.makedirs(d, exist_ok=True)

    # ------------------------------
    # Save Training Curves (CSV + PNG)
    # ------------------------------
    csv_file = os.path.join(training_dir, 'results.csv')
    loss = history.history['loss']
    val_loss = history.history.get('val_loss', [None] * len(loss))
    iou = history.history.get('iou_score', [None] * len(loss))
    val_iou = history.history.get('val_iou_score', [None] * len(loss))

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Training IOU', 'Validation IOU'])
        for epoch_idx, (l, vl, iu, viu) in enumerate(zip(loss, val_loss, iou, val_iou), start=1):
            writer.writerow([epoch_idx, float(l), float(vl) if vl is not None else None,
                             float(iu) if iu is not None else None,
                             float(viu) if viu is not None else None])
    print(f"Training results saved to CSV: {csv_file}")

    # Plot curves
    epochs_range = range(1, len(loss) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, 'y', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'r', label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, iou, 'y', label='Training IOU')
    plt.plot(epochs_range, val_iou, 'r', label='Validation IOU')
    plt.title('IOU over epochs')
    plt.xlabel('Epoch'); plt.ylabel('IOU'); plt.legend()

    plot_file = os.path.join(training_dir, "training_curves.png")
    plt.tight_layout(); plt.savefig(plot_file); plt.close()
    print(f"Training curves saved to: {plot_file}")

    # ------------------------------
    # Predictions & Visualization
    # ------------------------------
    visualize_predictions(model, X_val, y_val, n_samples=3, save_path=pred_dir)

    # ------------------------------
    # Full Evaluation Metrics
    # ------------------------------
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
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
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

            precision_i = TP / (TP + FP + 1e-8)
            recall_i = TP / (TP + FN + 1e-8)
            f1_i = 2 * precision_i * recall_i / (precision_i + recall_i + 1e-8)
            IoU = TP / (TP + FP + FN + 1e-8)
            numerator = (TP * TN) - (FP * FN)
            denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + 1e-8
            mcc_i = numerator / denominator

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


def visualize_predictions(model, X_val, y_val, n_samples=3, save_path=None):
    n_samples = min(n_samples, X_val.shape[0])  # avoid out-of-bounds
    n_classes = y_val.shape[-1] if y_val.ndim == 4 else len(np.unique(y_val))
    preds = model.predict(X_val)
    preds_labels = np.argmax(preds, axis=-1)
    y_val_labels = np.argmax(y_val, axis=-1) if y_val.ndim == 4 else y_val

    plt.figure(figsize=(12, n_samples * 4))
    for i in range(n_samples):
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

        # Flatten masks for metrics
        true_flat = true_mask.flatten()
        pred_flat = pred_mask.flatten()

        # Per-class metrics
        cm = confusion_matrix(true_flat, pred_flat, labels=list(range(n_classes)))
        precision = precision_score(true_flat, pred_flat, average='weighted', zero_division=0)
        recall = recall_score(true_flat, pred_flat, average='weighted', zero_division=0)
        f1 = f1_score(true_flat, pred_flat, average='weighted', zero_division=0)

        # Per-class IOU using keras MeanIoU
        IOU_keras = tf.keras.metrics.MeanIoU(num_classes=n_classes)
        IOU_keras.update_state(true_mask, pred_mask)
        iou_matrix = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
        per_class_iou = [iou_matrix[i,i] / (np.sum(iou_matrix[i,:]) + 1e-8) for i in range(n_classes)]

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

            cm_file = os.path.join(save_path, f"sample_{i}_confusion_matrix.png")
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=[f"Class {i}" for i in range(n_classes)],
                        yticklabels=[f"Class {i}" for i in range(n_classes)])
            plt.title(f"Sample {i} Confusion Matrix")
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.savefig(cm_file)
            plt.close()

    plt.tight_layout()
    if save_path:
        pred_fig_path = os.path.join(save_path, "sample_predictions.png")
        plt.savefig(pred_fig_path)
        print(f"Sample predictions saved to {pred_fig_path}")
    plt.close()




# ----------------------
# Training Pipeline
# ----------------------
class TrainPipeline:
    def __init__(self, config, version="v001", continue_training=False):
        self.debug = True
        self.config = config
        self.version = version
        self.continue_training = continue_training

        # Versioned paths
        self.train_images_path = os.path.join(config['paths']['training_data'], version, "images", "train")
        self.train_masks_path = os.path.join(config['paths']['training_data'], version, "masks", "train")
        self.test_images_path = os.path.join(config['paths']['training_data'], version, "images", "test")
        self.test_masks_path = os.path.join(config['paths']['training_data'], version, "masks", "test")

        self.model_path = os.path.join(config['paths']['models'], f"{config['training_params']['model_type']}_{version}")
        os.makedirs(self.model_path, exist_ok=True)

        self.n_classes = config['training_params']['n_classes']
        self.activation = config['training_params']['activation']
        self.batch_size = config['training_params']['batch_size']
        self.img_size = config['training_params']['img_size']
        self.BACKBONE = config['training_params']['model_type']
        self.model_type = config['training_params']['model_type']
        self.learning_rate = config['training_params'].get('learning_rate', 1e-4)

    def load_images_and_masks(self, images_dir, masks_dir):
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
            mask = TrainingLabelCreator().create_training_labels(mask)

            images.append(img)
            masks.append(mask)
        return np.array(images), np.array(masks)

    def preprocess_data(self, images, masks):
        labelencoder = LabelEncoder()
        n, h, w = masks.shape
        masks_encoded = labelencoder.fit_transform(masks.reshape(-1, 1)).reshape(n, h, w)
        masks_encoded = masks_encoded.astype(np.int32)  # <-- add this

        #masks_encoded = np.expand_dims(masks_encoded, axis=-1)
        #masks_cat = to_categorical(masks_encoded, num_classes=self.n_classes)
        X_images = sm.get_preprocessing(self.model_type)(images)
        return X_images, masks_encoded

    def build_model(self, dropout_rate=0.0):
        return sm.Unet(self.BACKBONE, encoder_weights='imagenet',
                       classes=self.n_classes, activation=self.activation,
                       dropout=dropout_rate)

    def _train_with_params(self, params, trial=None, epochs=50, save_models=True):
        if self.debug:
            epochs=2
        """
        Train or resume training with given params.
        Handles continue_training, checkpointing, early stopping, and optional Optuna pruning.
        Saves model weights and JSON training history only if save_models=True.
        Returns: model, history, X_val, y_val
        """
        # Paths
        model_file = os.path.join(self.model_path, f"{self.model_type}_final.keras")
        history_file = os.path.join(self.model_path, 'training_history.json')

        # Load data
        X_train, y_train = self.preprocess_data(
            *self.load_images_and_masks(self.train_images_path, self.train_masks_path))
        X_val, y_val = self.preprocess_data(
            *self.load_images_and_masks(self.test_images_path, self.test_masks_path))

        train_gen = CustomDataGenerator(X_train, y_train, self.batch_size, self.n_classes, augmentation=True)
        val_gen = CustomDataGenerator(X_val, y_val, self.batch_size, self.n_classes, augmentation=True)


        # DEBUG (small data)
        if self.debug:
            X_train, y_train = X_train[:5], y_train[:5]
            X_val, y_val = X_val[:2], y_val[:2]
            train_gen = CustomDataGenerator(X_train, y_train, batch_size=1, n_classes=self.n_classes, augmentation=False)
            val_gen = CustomDataGenerator(X_val, y_val, batch_size=1, n_classes=self.n_classes, augmentation=False)

        # Determine initial_epoch and load/build model
        initial_epoch = 0
        if self.continue_training:
            keras_files = [f for f in os.listdir(self.model_path) if f.endswith('.keras')]
            if len(keras_files) == 1:
                model = tf.keras.models.load_model(os.path.join(self.model_path, keras_files[0]), compile=False)
                print(f"Loaded model from {keras_files[0]}")
                if os.path.exists(history_file):
                    with open(history_file, 'r') as f:
                        json_history = json.load(f)
                    initial_epoch = len(json_history.get('loss', []))
                    print(f"Resuming from epoch {initial_epoch}")
            else:
                print("No single model found, starting from scratch.")
                model = self.build_model(dropout_rate=params.get("dropout_rate", 0.0))
        else:
            model = self.build_model(dropout_rate=params.get("dropout_rate", 0.0))

        # Compile model
        dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25]))
        focal_loss = sm.losses.CategoricalFocalLoss()
        total_loss = params.get("dice_loss_weight", 1.0) * dice_loss + params.get("focal_loss_weight", 1.0) * focal_loss
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=params["learning_rate"],
            decay=params.get("weight_decay", 0.0)
        )
        metrics = [sm.metrics.IOUScore(threshold=params.get("threshold_iou", 0.5)),
                   sm.metrics.FScore(threshold=params.get("threshold_fscore", 0.5))]
        model.compile(optimizer=optimizer, loss=total_loss, metrics=metrics)

        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        )]

        if save_models:
            epochs_path = os.path.join(self.model_path, "epochs")
            os.makedirs(epochs_path, exist_ok=True)

            # Save best model
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_path, f"best_model_{self.model_type}.keras"),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                mode='min',
                verbose=1
            ))

            # Save each epoch
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    epochs_path,
                    f"{self.model_type}_epoch{{epoch:02d}}_val{{val_loss:.4f}}.keras"
                ),
                save_freq='epoch',
                save_weights_only=False,
                verbose=1
            ))

        if trial is not None:
            callbacks.append(TFKerasPruningCallback(trial, "val_loss"))

        # Train
        history = model.fit(
            train_gen, validation_data=val_gen,
            epochs=initial_epoch + epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=1
        )

        # Save JSON history
        history_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in history.history.items()}
        history_dict['initial_epoch'] = initial_epoch
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                existing_history = json.load(f)
            for k, v in history_dict.items():
                existing_history.setdefault(k, [])
                if k != 'initial_epoch':
                    existing_history[k].extend(v)
            history_dict = existing_history
        with open(history_file, 'w') as f:
            json.dump(history_dict, f)

        return model, history, X_val, y_val

    def train_model(self, epochs=50):
        params = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.config["training_params"].get("weight_decay", 0.0),
            "dice_loss_weight": 0.0,
            "focal_loss_weight": 1.0,
            "threshold_iou": 0.5,
            "threshold_fscore": 0.5,
            "dropout_rate": 0.0,
        }

        model, history, X_val, y_val = self._train_with_params(params, epochs=epochs)
        log_training_results(model,history, X_val, y_val,save_path=self.model_path)

    def train_with_optuna_params(self):
        best_params_file = os.path.join(self.model_path, "best_params.json")
        with open(best_params_file, "r") as f:
            best_params = json.load(f)

        model, history, X_val, y_val = self._train_with_params(best_params)
        log_training_results(model,history, X_val, y_val,save_path=self.model_path)

    def optimize_hyperparameters(self, n_trials=35):
        if self.debug:
            n_trials =2
        def objective(trial):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                "dice_loss_weight": trial.suggest_float("dice_loss_weight", 0.1, 2.0),
                "focal_loss_weight": trial.suggest_float("focal_loss_weight", 0.1, 2.0),
                "threshold_iou": trial.suggest_float("threshold_iou", 0.3, 0.7),
                "threshold_fscore": trial.suggest_float("threshold_fscore", 0.3, 0.7),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
            }
            model, history, _, _ = self._train_with_params(params, trial=trial, save_models=False)
            return history.history["val_loss"][-1]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params_file = os.path.join(self.model_path, "best_params.json")
        with open(best_params_file, "w") as f:
            json.dump(study.best_params, f)
        print("Best Optuna parameters saved:", best_params_file)


# ----------------------
# Entry Point
# ----------------------
if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    parser = argparse.ArgumentParser(description="Train segmentation model with optional Optuna support")
    parser.add_argument("--version", type=str, default="v001", help="Version for training outputs")
    parser.add_argument("--optuna", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--optuna-load", action="store_true", help="Train using best Optuna parameters")
    parser.add_argument("--continue-training", action="store_true", help="Continue training from a saved checkpoint")
    args = parser.parse_args()

    base_path = os.path.join(PROJECT_ROOT, "config", "base_config.yaml")
    train_path = os.path.join(PROJECT_ROOT, "config", "training_config.yaml")
    config = config_helper.ConfigLoader.load_config(base_path, train_path)

    trainer = TrainPipeline(config, version=args.version, continue_training=args.continue_training)

    if args.optuna:
        trainer.optimize_hyperparameters()
    elif args.optuna_load:
        trainer.train_with_optuna_params()
    else:
        trainer.train_model()
