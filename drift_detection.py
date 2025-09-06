# drift_detection.py
import os
import pandas as pd
import mlflow
from scipy.stats import ks_2samp

def detect_and_log_drift(train_csv, new_csv, out_dir="artifacts"):
    """
    Detects drift between training and new datasets using KS test.
    Logs results into MLflow and saves a CSV report locally.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load datasets
    df_train = pd.read_csv(train_csv)
    df_new = pd.read_csv(new_csv)

    # Run KS test per column
    results = []
    for col in df_train.columns:
        if col in df_new.columns:
            stat, p_value = ks_2samp(df_train[col], df_new[col])
            drifted = p_value < 0.05
            results.append({"feature": col, "stat": stat, "p_value": p_value, "drifted": drifted})

    rpt = pd.DataFrame(results)

    # Save report locally
    out_csv = os.path.join(out_dir, "drift_report.csv")
    rpt.to_csv(out_csv, index=False)

    # Log safe metrics
    num_drifted = rpt["drifted"].sum()
    mlflow.log_metric("num_features_drifted", int(num_drifted))
    mlflow.log_metric("total_features", int(len(rpt)))
    mlflow.log_metric("drift_fraction", float(num_drifted) / max(1, len(rpt)))

    # log drift stats for each feature
    for _, row in rpt.iterrows():
        safe_name = f"drift_pvalue_{row['feature']}".replace(" ", "_")
        mlflow.log_metric(safe_name, row["p_value"])

    mlflow.log_artifact(out_csv)

    print(f"Drift detection complete. {num_drifted}/{len(rpt)} features drifted.")
    return rpt
