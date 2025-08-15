# data_ingestion.py
import os
import joblib
import pandas as pd
import numpy as np
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
                         out_dir="."):
    """
    Loads csv, scales features with MinMaxScaler, splits to train/valid/test,
    saves scaler and split CSVs/npz files for later use.
    Returns: X_train, X_valid, X_test, y_train, y_valid, y_test, scaler, feature_names
    """
    os.makedirs(out_dir, exist_ok=True)
    df = load_csv(csv_path)
    # keep original feature names for later plotting
    feature_names = [c for c in df.columns if c not in [id_col, target_col]]

    X = df[feature_names].copy()
    y = df[target_col].copy()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # First split test
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X_scaled, y.values, test_size=test_size, random_state=random_state, stratify=y
    )
    # Now split train_valid into train and valid
    # valid_size is fraction of original dataset; convert to fraction of train_valid
    valid_frac_of_train_valid = valid_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=valid_frac_of_train_valid,
        random_state=random_state, stratify=y_train_valid
    )

    # Save scaler and CSVs for convenience
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    np.savez_compressed(os.path.join(out_dir, "splits.npz"),
                        X_train=X_train, X_valid=X_valid, X_test=X_test,
                        y_train=y_train, y_valid=y_valid, y_test=y_test,
                        feature_names=np.array(feature_names))
    # also save CSVs (for sanity)
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train[target_col] = y_train
    df_train.to_csv(os.path.join(out_dir, "train.csv"), index=False)

    df_valid = pd.DataFrame(X_valid, columns=feature_names)
    df_valid[target_col] = y_valid
    df_valid.to_csv(os.path.join(out_dir, "valid.csv"), index=False)

    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test[target_col] = y_test
    df_test.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print(f"Saved scaler.pkl and splits.npz to {out_dir}")
    return X_train, X_valid, X_test, y_train, y_valid, y_test, scaler, feature_names

if __name__ == "__main__":
    # Run as script example (in Colab)
    X_train, X_valid, X_test, y_train, y_valid, y_test, scaler, feature_names = preprocess_and_split(
        "diabetes.csv", out_dir="artifacts"
    )
