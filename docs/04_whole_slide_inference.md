
### 4. Whole-Slide / Large Image Inference  
**Script**: `whole_slide_prediction.py`

---

#### Purpose  
Perform scalable, inference-only semantic segmentation and refinement on whole-slide or large immunohistochemistry (IHC) images using a pre-trained deep learning model.

---

#### Description  
The `whole_slide_prediction.py` script implements an inference-only pipeline designed to efficiently segment very large histopathology images while preserving biologically meaningful tissue structures. A coarse-to-fine prediction strategy is used to balance scalability and boundary accuracy.

---

#### Key Strategy  
- Initial coarse segmentation on downsampled whole-slide images
- Post-processing of predicted masks to:
  - Retain biologically meaningful connected tissue regions
  - Preserve mixed epidermis–keratin components
  - Suppress small, spurious epidermal predictions
- Automatic identification of dominant tissue regions
- Full-resolution cropping and re-segmentation of dominant regions
- Reintegration of refined predictions into corrected full-resolution masks

---

#### Features  
- Multi-class segmentation of epidermis and keratin
- Flexible input sources:
  - Image URLs loaded directly from a local HPA SQLite database
  - Local folder mode for testing, debugging, or external datasets
- Automatic resume capability using a persistent `last_processed_id`
- Reproducible, versioned output structure linked to the specific model used
- Comprehensive visualization outputs and spatial metadata

---

#### Outputs  
- Full-resolution corrected segmentation masks (`.npz`)
- Cropped region predictions with spatial metadata
- JSON files describing crop geometry and original image dimensions
- Multi-panel visualization figures summarizing each prediction

---

#### Usage  
```bash
# Database-driven inference
python3 -m scripts.whole_slide_prediction \
    --model path/to/model.keras \
    --mode database \
    --version v001

# Local folder inference
python3 -m scripts.whole_slide_prediction \
    --model path/to/model.keras \
    --mode folder \
    --input_folder path/to/images \
    --version v001
```

---

#### Configuration

Pipeline behavior is controlled via:

* `base_config.yaml`
* `prediction_config.yaml`

These files define paths, database locations, and model-related parameters.

---

#### Integration Flow

```plaintext
data_preprocessing scripts
       │
       ▼
Cropped H&E / mask data
       │
       ▼
train.py
(segmentation model training)
       │
       ▼
Trained model (.keras)
       │
       ▼
whole_slide_prediction.py
(whole-slide IHC inference + refinement)
       │
       ▼
Corrected full-resolution masks
+ cropped predictions
+ spatial metadata
+ visualization summaries
```

*(The HPA database construction pipeline runs independently to supply image URLs and metadata.)*

---

#### Publication / Figure Summary

> Whole-slide IHC images were segmented using a pre-trained deep learning model in an inference-only pipeline. Large images were first downsampled for coarse prediction, followed by region-focused cropping and re-segmentation to refine tissue boundaries. Post-processing preserved biologically meaningful epidermis and keratin regions while suppressing spurious predictions. Corrected masks, spatial metadata, and visualization summaries were stored in a reproducible, versioned structure.

---

#### Methods

Whole-slide and large immunohistochemistry (IHC) images were segmented using a custom inference-only deep learning pipeline. A pre-trained ResNet-based U-Net model was applied to downsampled images to obtain an initial coarse segmentation. Predicted masks were post-processed to retain biologically meaningful connected tissue regions, preserve mixed epidermis–keratin components, and suppress small spurious predictions.

Dominant predicted tissue regions were automatically identified, mapped back to full image resolution, cropped, and re-segmented to improve boundary accuracy. The refined predictions were reintegrated into full-resolution masks. Images were processed either directly from URLs stored in a local Human Protein Atlas (HPA) SQLite database or from local folders for testing and validation. All outputs were stored in a versioned directory structure linked to the specific model used, ensuring reproducible large-scale inference.