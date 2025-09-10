#!/bin/bash
set -e

echo "Running in mode: $RUN_MODE"

case "$RUN_MODE" in
  "training")
    echo "Starting training script (train_optuna.py)..."
    python3 scripts/training/train_optuna.py
    ;;
  "training_load")
    echo "Starting training load script (train_optuna_load.py)..."
    python3 scripts/training/train_optuna_load.py
    ;;
  "epidermis_analysis")
    echo "Starting epidermis analysis script (ihc_cellpose.py)..."
    python3 scripts/epidermis_analysis/ihc_cellpose.py
    ;;
  "dataprep")
    echo "Starting data preprocessing script (crop_data.py)..."
    python3 scripts/data_preprocessing/crop_data.py
    ;;
  "prediction")
    echo "Starting prediction script (whole_slide_prediction.py)..."
    python3 scripts/prediction/whole_slide_prediction.py
    ;;
  "database")
    echo "Starting database build script (build_hpa_database.py)..."
    python3 scripts/database/build_hpa_database.py
    ;;
  *)
    echo "Unknown RUN_MODE: $RUN_MODE"
    exit 1
    ;;
esac
