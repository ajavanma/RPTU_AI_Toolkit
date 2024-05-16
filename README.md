# RPTU AI Toolkit
A toolkit for preprocessing, training and inference

## Getting Started

1. Create and activate a conda environment:
   ```sh
   conda create --name mink python=3.8
   conda activate mink

2. Install Poetry:
   ```sh
   curl -sSL https://install.python-poetry.org | python3 -

3. Install project dependencies:
poetry install

4. Set config parameters in config/config.yaml file

5. poetry run python src/main.py


set config parameters in config.yaml file   
conda activate mink  
python main.py   

# File hierarchy
data/     &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp;  &nbsp;   &nbsp;  &nbsp;   &nbsp;   # Raw and processed data (pcd, asc, ply), mapping dictionaries (yaml)  
logs/  &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp; &nbsp;   &nbsp;   &nbsp;  &nbsp;     # Logs and reports  
models/   &nbsp;   &nbsp;   &nbsp;   &nbsp;  &nbsp;   &nbsp;       # Trained models  
notebooks/  &nbsp;   &nbsp;   &nbsp;     # Jupyter notebooks for exploration and visualization (sanity check of raw and processed data)  
src/    &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp;    &nbsp;   &nbsp;   &nbsp;   &nbsp;    # Source code  
├── __init__.py  
├── config/   
│   ├── config.yaml  
│   ├── preprocessing.yaml  
│   ├── training.yaml  
│   └── inference.yaml  
├── data/   
│   └── preprocessing.py  
├── models/  
│   └── model.py  
├── training/   
│   ├── train.py  
│   └── evaluate.py  
├── inference/  
│   └── infer.py  
├── utils/   
│   └── utils.py   
└── main.py       
tests/       

 
# Requirements
MinkowskiEngine: calculation of sparse tensors  
o3d: manipulation of points (coords), colors and normals  

# Program structure
Data Preparation
1. Raw Data Analysis: Generate report of each asc and pcd file, check for skewed distribution.
2. Preprocessing:
Data Preparation: Feature selection, data cleansing, transforming, outlier removal, scaling (normalization/standardization).
Feature Engineering: Imputation (managing missing data), one-hot encoding of categorical features.
3. Reports
Training and Evaluation
Iteration (hyperparameter tuning)
Test and Verification
Visualizing
