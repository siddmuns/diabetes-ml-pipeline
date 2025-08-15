# Diabetes ML Pipeline (Colab / Local)

This repository contains an end-to-end Diabetes prediction ML pipeline implemented for Google Colab or local execution. The pipeline covers:

- Data ingestion and preprocessing (MinMaxScaler)
- Hyperparameter tuning using Optuna
- Experiment tracking and model registration using MLflow
- Model training (Gradient Boosting Classifier)
- Batch inference on unseen `diabetes2.csv`
- Manual data drift detection (KS test) and visualization
- Evaluation plots (ROC, confusion matrix, feature importance) logged to MLflow

## Structure

- `data_ingestion.py` - load, scale, split, and save datasets (train/valid/test); saves `scaler.pkl` and `splits.npz`
- `tune_and_train.py` - runs Optuna HPO (logs each trial to MLflow), retrains final model, logs final run and model
- `batch_inference.py` - loads scaler + model, runs predictions on `diabetes2.csv`, outputs `predictions.csv`
- `drift_detection.py` - compares `diabetes.csv` vs `diabetes2.csv` via KS tests and logs plots to MLflow
- `visualize_and_evaluate.py` - creates ROC, confusion matrix, and feature importance plots and logs them
- `pipeline_notebook.ipynb` - Colab notebook orchestration (cells provided in repository README)
- `requirements.txt` - Python dependencies

## Quick start (Colab)

1. Upload `diabetes.csv` and `diabetes2.csv` to Colab.
2. Install dependencies:
   ```bash
   !pip install -q -r requirements.txt
