Project structure:

/ (Root)
│
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   ├── v001/
│   │   ├── training/
│   │   ├── predictions/  
│   │   ├── cellpose/
│   │   └── layer_analysis/
│
├── config/
│   ├── README.md
│   ├── base_config.yaml           
│   ├── dataprep_config.yaml       
│   ├── epidermis_analysis_config.yaml   
│   ├── hpa_database_config.yaml  
│   ├── training_config.yaml       
│   └── prediction_config.yaml     
│   
├── environment/
│   ├── assets/
│        └── ...
│   ├── requirements.txt          
###│   ├── Dockerfile            
###│   └── .env                  
###│   └── entrypoint.sh           
│
├── resources/
│   ├── final.db
│   ├── normal_expression_Skin_Any.tsv
│
├── scripts/
│   ├──config_helper.py
│   ├── data_preprocessing/
│   │   ├── automated_cropping.py      
│   │   ├── manual_cropping.py
│   │   └── crop_data.py                
│   │
│   ├── database/
│   │   └── build_hpa_database.py      
│   │
│   ├── training/
│   │   └── train.py     
│   ├── epidermis_analysis/
│   │   ├── cell_layers_database.py             
│   │   └── ihc_cellpose.py            
│   │
│   └── prediction/
│       └── whole_slide_prediction.py

│
├── models/
│   └── f"{config['training_params']['model_type']}_{version}"/
│   │   ├── epochs/
│   │   ├── evaluation/
│   │   ├── training_curves/
│   │   └── sample_predictions/
│
├── README.md
├── LICENSE
└── .gitignore

run scripts:

pip install -r environment/requirements.txt

# cropping
py -m scripts.data_preprocessing.crop_data

# human Protein Atlas XML scrape
py -m scripts.database.build_hpa_database

# standard training, continuation, and Optuna hyperparameter search
py -m scripts.training.train --version v001 --optuna
py -m scripts.training.train --version v001 --optuna-load
py -m scripts.training.train --version v001 --optuna-load --continue-training


#--linux
# prediction
python3 -m scripts.prediction.whole_slide_prediction \
  --version v001 \
  --model /mnt/c/Users/kfayd01/Downloads/resnet152_epoch63_val0.1902_cont.keras \
  --mode database 	# Choice between 'database' and 'folder'
  --input_folder … 	# Needed if mode is 'folder'


# cellpose
python3 -m scripts.epidermis_analysis.ihc_cellpose \
  --version v001 

# layers analysis
python3 -m scripts.epidermis_analysis.cell_layers_database \
  --version v001 \
  --gene TP63		# Optional, to filter for gene




---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code in your research, please cite:

**Name**, Contributor Name, *Title of Paper*, Journal/Conference, Year. [DOI or preprint link if available]

---

## Acknowledgements

Thanks to **Contributor Name** for testing and feedback on the code.  

---

