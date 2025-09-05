# batch_inference.py
import os
import joblib
import pandas as pd
import numpy as np

def batch_predict(model_path, scaler_path, input_csv, output_csv, id_col="PatientID"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df = pd.read_csv(input_csv)
    feature_cols = [c for c in df.columns if c != id_col and c != "Diabetic"]
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else np.nan

    out = df[[id_col]].copy() if id_col in df.columns else pd.DataFrame(index=df.index)
    out["Predicted_Diabetic"] = preds
    out["Predicted_Prob"] = probs
    out.to_csv(output_csv, index=False)
    print(f"Wrote predictions to {output_csv}")
    return out

if __name__ == "__main__":
    batch_predict("artifacts/gb_final_model.pkl", "artifacts/scaler.pkl", "diabetes2.csv", "artifacts/predictions.csv")
