Project structure:

/ (Root)
│
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   ├── v001/
│   │   ├── training/
│   │   └── predictions/  
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
│   ├── Dockerfile                 
│   └── .env                      
│   └── entrypoint.sh             
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
