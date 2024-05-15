### RPTU AI Toolkit
A toolkit for preprocessing, training and inference

## Getting started
set config parameters in config.yaml file   
conda activate mink  
python main.py   

# File hierarchy
data/       raw and processed data (pcd, asc, ply), mapping dictionaries (yaml)  
logs/       logs and reports
models/     trained models  
notebooks/  Jupyter notebooks for exploration and visualization (sanity check of raw and processed data)  
src/        source code for feature engineering in preprocessing, model structure, dataset generator, training and testing implementation   
utils/      utility functions  

# Requirements
MinkowskiEngine: calculation of sparse tensors  
o3d: manipulation of points (coords), colors and normals  
PyTorch Lightning: high level pytorch framework    

# Program structure
Data Preparation   
    Raw Data Analysis: generate report of each asc and pcd file, check for skewd distribution  
    Preprocessing:  
        Data Preparation: Feature Selection, data cleansing, transforming, outlier removal, Scaling(normaliation/standardization)  
        Feature Engineering: imputation(managing missing data), one-hot encoding of categorical features  
    Reports    
Training and Evaluation  
Iteration (hyperparameter tuning)  
Test and Verification  
Visualizing  
Inference   
