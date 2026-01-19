
### 2. HPA Database Construction  
**Script**: `build_hpa_database.py`

---

#### Purpose  
Construct a local SQLite database from Human Protein Atlas (HPA) TSV and XML files for reproducible downstream analysis.

---

#### Description  
The `build_hpa_database.py` script builds a local Human Protein Atlas (HPA) database by loading gene and antibody information from a TSV file, retrieving corresponding XML records from the HPA website, extracting tissue- and cell-level expression data, and storing validated results in an SQLite database.

---

#### Main Steps  
```plaintext
1. Load HPA gene/antibody TSV file
2. Retrieve corresponding XML files for each gene from HPA
3. Parse XML to extract:
   - Gene name and gene ID
   - Antibody ID
   - Tissue-specific expression (focused on skin)
   - IHC image URLs
   - Patient metadata (sex, age, patient ID)
   - Tissue-cell specific annotations (cell type, staining, intensity, quantity, location)
   - Antibody reliability / verification
4. Validate and deduplicate extracted records
5. Store unique entries in SQLite database (final.db / table: elGenes)
````

---

#### Features

* Configuration-driven TSV path selection via `base_config.yaml` and `hpa_database_config.yaml`
* Fallback file picker if TSV path is missing or invalid
* Safe handling of missing or incomplete XML fields
* Deduplication to ensure unique gene–antibody–tissue–image records
* Progress bar (`tqdm`) for real-time execution feedback

---

#### Data Storage

* Database: `final.db`
* Table: `elGenes`
* Each row represents a unique gene–antibody–tissue–image combination, including patient and tissue-cell metadata.

---

#### Publication / Figure Summary

> A local Human Protein Atlas database was constructed by parsing TSV gene and antibody information and retrieving corresponding XML records from the HPA. Tissue-specific protein expression (skin), immunohistochemistry (IHC) image URLs, patient metadata, cell-type-specific annotations, and antibody reliability information were extracted and stored in an SQLite database for downstream analysis.

---

#### Methods

A local Human Protein Atlas (HPA) database was constructed using a custom Python pipeline. A TSV file provided by HPA, containing gene identifiers and antibody information, was first loaded. For each gene, the corresponding XML file was downloaded from the HPA website, providing structured data on tissue-specific protein expression, immunohistochemistry (IHC) image URLs, patient metadata (sex, age, patient ID), and antibody reliability annotations.

The XML files were parsed programmatically to extract expression levels, cell-type-specific staining characteristics, and image–patient mappings, with a focus on skin tissue. Missing or incomplete fields were handled gracefully to maintain data integrity, and duplicate entries were filtered. All unique records were stored in a local SQLite database (`final.db`) in a table representing gene–antibody–tissue–image associations.

A progress indicator was included to provide real-time feedback during database construction. This pipeline enables automated, reproducible assembly of HPA data for downstream analysis of tissue-specific protein expression and immunohistochemistry experiments.
