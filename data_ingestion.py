# data_ingestion.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_csv(path):
    return pd.read_csv(path)

def preprocess_and_split(csv_path,
                         id_col="PatientID",
                         target_col="Diabetic",
                         test_size=0.20,
                         valid_size=0.10,
                         random_state=42,
                         out_dir="artifacts"):
    """
    Loads csv, scales features with MinMaxScaler, splits to train/valid/test,
    saves scaler and split arrays for later use.
    Returns: X_train, X_valid, X_test, y_train, y_valid, y_test, scaler, feature_names
    """
    os.makedirs(out_dir, exist_ok=True)
    df = load_csv(csv_path)

    feature_names = [c for c in df.columns if c not in [id_col, target_col]]
    X = df[feature_names].copy()
    y = df[target_col].copy().values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # First split out test
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Then split train_valid into train and valid
    valid_frac_of_train_valid = valid_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=valid_frac_of_train_valid,
        random_state=random_state, stratify=y_train_valid
    )

    # Save artifacts
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    np.savez_compressed(os.path.join(out_dir, "splits.npz"),
                        X_train=X_train, X_valid=X_valid, X_test=X_test,
                        y_train=y_train, y_valid=y_valid, y_test=y_test,
                        feature_names=np.array(feature_names))

    # Save CSVs for convenience (optional)
    pd.DataFrame(X_train, columns=feature_names).assign(**{target_col: y_train}).to_csv(os.path.join(out_dir, "train.csv"), index=False)
    pd.DataFrame(X_valid, columns=feature_names).assign(**{target_col: y_valid}).to_csv(os.path.join(out_dir, "valid.csv"), index=False)
    pd.DataFrame(X_test,  columns=feature_names).assign(**{target_col: y_test}).to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print(f"Saved scaler.pkl and splits.npz to {out_dir}")
    return X_train, X_valid, X_test, y_train, y_valid, y_test, scaler, feature_names

if __name__ == "__main__":
    preprocess_and_split("diabetes.csv", out_dir="artifacts")
