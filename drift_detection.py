# drift_detection.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import mlflow

def detect_and_log_drift(ref_csv, new_csv, id_col="PatientID", target_col="Diabetic", out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    df_ref = pd.read_csv(ref_csv)
    df_new = pd.read_csv(new_csv)

    # choose numeric features excluding ID/target
    features = [c for c in df_ref.columns if c not in [id_col, target_col]]

    drift_report = {}
    for feat in features:
        s1 = df_ref[feat].dropna()
        s2 = df_new[feat].dropna()
        stat, pvalue = ks_2samp(s1, s2)
        drift_report[feat] = {"ks_stat": float(stat), "p_value": float(pvalue)}

        # plot histograms
        fig, ax = plt.subplots()
        ax.hist(s1, bins=25, alpha=0.5, label="ref")
        ax.hist(s2, bins=25, alpha=0.5, label="new")
        ax.set_title(f"{feat} (KS={stat:.3f}, p={pvalue:.3f})")
        ax.legend()
        fig_path = os.path.join(out_dir, f"drift_{feat}.png")
        fig.savefig(fig_path)
        plt.close(fig)

    # save report
    report_path = os.path.join(out_dir, "drift_report.json")
    with open(report_path, "w") as f:
        json.dump(drift_report, f, indent=2)

    # log to mlflow as artifacts and metrics in a run
    mlflow.set_experiment("Diabetes_Pipeline")
    with mlflow.start_run(run_name="Drift_Report"):
        for feat, vals in drift_report.items():
            mlflow.log_metric(f"ks_stat_{feat}", vals["ks_stat"])
            mlflow.log_metric(f"p_value_{feat}", vals["p_value"])
            img_path = os.path.join(out_dir, f"drift_{feat}.png")
            if os.path.exists(img_path):
                mlflow.log_artifact(img_path, artifact_path=f"drift_plots/{feat}")
        mlflow.log_artifact(report_path, artifact_path="drift_report")

    print(f"Drift report saved to {report_path}")
    return drift_report

if __name__ == "__main__":
    rpt = detect_and_log_drift("diabetes.csv", "diabetes2.csv", out_dir="artifacts")
    print(rpt)
