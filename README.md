# malaria-spatiotemporal-gru

# Spatio-Temporal Modeling of Malaria Incidence using GRU

This repository contains the data and code used in the paper:

**"Modeling and analyzing non-linear spatio-temporal patterns of malaria incidence in sub-Saharan Africa using recurrent neural networks"**

## Overview

This study analyzes malaria incidence dynamics across sub-Saharan Africa using climate variables and temporal modeling. The approach combines exploratory data analysis with machine learning and deep learning models, including GRU (Gated Recurrent Units), Random Forest, and ARIMA. The objective is to model non-linear spatio-temporal relationships and evaluate predictive performance across regions.

## Repository Structure

data/ → final_dataset.csv, dataset_with_lags.csv  
scripts/ → 01_feature_engineering.py, 02_gru_model.py, 03_random_forest.py, 04_arima.py, 05_eda.py, 06_observed_vs_predicted.py, 07_error_analysis.py  
requirements.txt  
README.md  

## Data

dataset_with_lags.csv is the final dataset used for all modeling tasks. It includes temperature, precipitation, humidity, lagged precipitation, and malaria incidence.  
final_dataset.csv contains the merged dataset (incidence + climate) before lag feature construction.

## Note on Data

The dataset used in this repository is derived from the Malaria Atlas Project (MAP) and NASA POWER climate data. Preprocessing steps such as data collection and cleaning are not included; the final dataset used for modeling is provided directly for reproducibility.

## Installation

Install required dependencies using:

pip install -r requirements.txt

## How to Reproduce Results

Run scripts in the following order:

1. 01_feature_engineering.py  
2. 02_gru_model.py  
3. 03_random_forest.py  
4. 04_arima.py  
5. 05_eda.py  
6. 06_observed_vs_predicted.py  
7. 07_error_analysis.py  

## Models

GRU (proposed model): uses sequences of length 5 with climate variables, lagged variables, and past incidence, capturing temporal dependencies and non-linear relationships.  
Random Forest: non-linear baseline using the same feature set as GRU but without explicit temporal modeling.  
ARIMA: classical time-series model applied independently per region, using only past incidence.

## Output

The scripts generate model performance metrics (MAE and RMSE), observed vs predicted plots, error distribution across regions, and temporal stability analysis.

## Reproducibility

All models are trained and evaluated using the same temporal split: training data up to 2016, validation from 2017 to 2020, and test data from 2021 onwards.

## Notes

Normalization is performed using statistics computed exclusively from the training period. GRU results are reported in both normalized and original scales. Error analysis is conducted in the original scale (cases per 1,000 population).

## Contact

For questions or further information:  
J. Salvador Sánchez (sanchez AT uji.es)

## License

This repository is provided for academic and research purposes.
