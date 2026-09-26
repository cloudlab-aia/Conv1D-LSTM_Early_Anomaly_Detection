Early Anomaly Detection in Time Series Using a Hybrid Conv1D-LSTM Model with Dynamic Thresholding
This repository contains the datasets, source code and experimental results associated with the paper:
Early Anomaly Detection in Time Series Using a Hybrid Conv1D-LSTM Model with Dynamic Thresholding: A Benchmark-Based Study

Overview
The paper proposes a hybrid predictive-adaptive framework for early anomaly detection in time series. The approach combines a multi-step Conv1D-LSTM predictor with an error-based detector that generates alarms from the discrepancy between observed values and model predictions.
The detection stage uses either dynamic or static thresholding, depending on the characteristics of each dataset. The evaluation follows an anticipation-aware policy that distinguishes between nominal regions, pre-event anticipation windows and effective anomaly regions.
The experimental validation includes three main scenarios:
- Early detection of COVID-19 outbreaks in Italian regions.
- Anomaly detection in the Power Demand dataset.
- Anomaly detection in the Space Shuttle dataset.
  
Repository content

The repository is organised as follows:
- datasets/: datasets used in the experimental validation.
- code/: source code used to train the models, generate predictions, apply the detection thresholds and compute the experimental results.
- results/: generated figures, plots and result files associated with the experiments reported in the paper.
  
Datasets

The datasets folder includes the time series used in the experiments:
- covid_italy/: data used for the early detection of COVID-19 outbreaks in Italian regions.
- power_demand/: benchmark time series corresponding to the Power Demand dataset.
- space_shuttle/: benchmark time series corresponding to the Space Shuttle dataset.
These datasets are included to support reproducibility of the experimental results reported in the paper.

Code

The code folder contains the scripts used to reproduce the experimental workflow described in the paper. The code covers the main stages of the proposed methodology:
- Time series preprocessing.
- Conv1D-LSTM model training.
- Multi-step prediction.
- Prediction error computation.
- Dynamic or static thresholding.
- Alarm generation.
- Evaluation using anticipation-aware metrics.
  
Results

The results folder contains the main experimental outputs generated during the validation of the proposed method, including plots, alarm visualisations and figures used in the manuscript.

Authors
- Rafael Rodrigo Guillén
- Higinio Mora Mora
- Jorge Azorín López
  
Department of Computer Technology and Computation
University of Alicante, Spain
