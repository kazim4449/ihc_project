
# Configuration Files for Skin IHC / Epidermis Analysis Pipeline

This folder contains all YAML configuration files used across the skin image analysis pipeline, including preprocessing, model training, and epidermis layer analysis. Each file defines paths, parameters, and other settings used by the scripts.  

---

## Files Overview

### 1. `base_config.yaml`
**Purpose:** Base configuration for all scripts.  
**Used by:** All scripts  
**Contents:**  
- General paths for raw and processed data, resources, and models.  
- UI/colors settings.  
- Environment assets paths (e.g., logos, arrows).  

---

### 2. `dataprep_config.yaml`
**Purpose:** Configuration for data preparation and image loading.  
**Used by:** data_preprocessing scripts.  
**Contents:**  
- Paths for H&E ROI images, masks, and coordinates.  
- Paths to UI assets (arrows, logo).  
- Background/foreground colors for visualization.  

---

### 3. `hpa_database_config.yaml`
**Purpose:** Configuration for loading the HPA dataset.  
**Used by:** `build_hpa_database.py`.  
**Contents:**  
- Path to HPA TSV file with skin expression data.  
- Path to SQLite database storing processed results.  
- Image/mask paths used for downstream processing.  
- Colors inherited from `base_config.yaml`.  

---

### 4. `training_config.yaml`
**Purpose:** Parameters for deep learning model training.  
**Used by:** `train.py`.  
**Contents:**  
- Model backbone selection (`resnet34`, `resnet152`).  
- Number of classes and activation function.  
- Learning rate, batch size, image size.  

---

### 5. `prediction_config.yaml`
**Purpose:** Configuration for running predictions on new data.  
**Used by:** `whole_slide_prediction.py`.  
**Contents:**  
- Paths to trained models.  
- Database path to retrieve IHC images.  
- Training parameters imported from `training_config.yaml`.  

---

### 6. `epidermis_analysis_config.yaml`
**Purpose:** Configuration for epidermis layer analysis from IHC images.  
**Used by:** `ihc_cellpose.py`, `cell_layers_database.py`.  
**Contents:**  
- Paths to IHC masks (`masks_ihc`) and cell masks (`masks_cells`).  
- Database path for retrieving IHC images and storing calculated epidermis layer metrics.  

---

## Notes

- Most YAML files include `base_config.yaml` to inherit common paths and settings.  
- Use the `${variable}` syntax to refer to paths defined in other config files.  
- Versioned outputs (e.g., `v001`) are defined in folder names and used in analysis scripts.  
- Colors, paths, and model settings can be overridden in specific configs.  

---

## Example Usage

```bash
python3 -m scripts.epidermis_analysis.cell_layers_database --version v001 --gene TP63