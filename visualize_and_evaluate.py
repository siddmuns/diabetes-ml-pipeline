# visualize_and_evaluate.py
import os
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay

def make_plots_and_log(model, X_test, y_test, feature_names, run_name="Final_Plots", nested=True):
    mlflow.set_experiment("Diabetes_Pipeline")
    with mlflow.start_run(run_name=run_name):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # ROC
        fig1, ax1 = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax1)
        ax1.set_title("ROC Curve")
        roc_path = "plots/roc_curve.png"
        os.makedirs("plots", exist_ok=True)
        fig1.savefig(roc_path)
        plt.close(fig1)
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig2, ax2 = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax2)
        ax2.set_title("Confusion Matrix")
        cm_path = "plots/confusion_matrix.png"
        fig2.savefig(cm_path)
        plt.close(fig2)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Feature importances
        importances = model.feature_importances_
        order = np.argsort(importances)
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.barh(np.array(feature_names)[order], importances[order])
        ax3.set_title("Feature Importances")
        ax3.set_xlabel("importance")
        fi_path = "plots/feature_importances.png"
        fig3.savefig(fi_path)
        plt.close(fig3)
        mlflow.log_artifact(fi_path, artifact_path="plots")

    print("Plots logged to MLflow.")

if __name__ == "__main__":
    print("Call make_plots_and_log(model, X_test, y_test, feature_names) from your notebook.")
