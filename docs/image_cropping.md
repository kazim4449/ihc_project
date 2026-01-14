
### 1. Image Cropping Pipeline  
**Scripts**: `crop_data.py`, `manual_cropping.py`, `automated_cropping.py`

---

#### Description  
This module provides a semi-automated image cropping pipeline for extracting regions of interest (ROIs) from H&E, mask, and IHC images. It supports both interactive manual cropping and reproducible automated batch processing.

---

#### Features  

**Manual Cropping**
- Launches an interactive GUI for selecting ROIs
- Users can resize a movable square selection
- Square regions can be subdivided into multiple subregions
- Cropped images are saved together with coordinate metadata for reproducibility

**Automated Cropping**
- Applies previously saved coordinate files to batch-process images
- Ensures consistent cropping across H&E, mask, and IHC modalities
- Cropped outputs use clear, consistent naming conventions
- Cropped images remain linked to their original coordinate metadata

---

#### Usage  
- `crop_data.py` – Launch the main menu GUI  
- `manual_cropping.py` – Open the manual cropping interface  
- `automated_cropping.py` – Apply saved coordinates to batch-crop images  

---

#### Configuration  
The pipeline uses YAML configuration files to ensure reproducibility and flexibility:
- `base_config.yaml`
- `dataprep_config.yaml`

These files control paths, file types, UI colors, and other processing parameters.

---

#### Data Management  
Cropped images and corresponding coordinate files are stored in organized directory structures, enabling reproducible downstream analysis across datasets.

---

#### Publication / Figure Summary  
> Cropped regions of H&E, mask, and IHC images were generated using a semi-automated pipeline. Manual selection defined regions of interest (ROIs), which were then used for automated batch cropping to ensure consistent data preparation across all images.

---

#### Methods  

To prepare H&E, mask, and IHC images for downstream analysis, a semi-automated image cropping pipeline was developed. The pipeline provides both manual and automated modes.  

In the manual cropping mode, a graphical user interface allows users to select regions of interest from individual images. A movable square can be resized, subdivided into multiple subregions, and saved along with coordinate metadata to ensure reproducibility.  

In the automated cropping mode, pre-defined coordinates obtained during manual cropping are applied to batch-process multiple images. Cropped regions are saved using consistent naming conventions and remain linked to their original images and coordinate files.  

The pipeline relies on YAML configuration files for path management, file types, and user interface parameters, ensuring consistent and reproducible data preparation across datasets.
