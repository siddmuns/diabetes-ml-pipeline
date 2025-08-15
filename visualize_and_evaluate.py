# visualize_and_evaluate.py
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay

def make_plots_and_log(model, X_test, y_test, feature_names, run_name="Final_Plots"):
    mlflow.set_experiment("Diabetes_Pipeline")
    with mlflow.start_run(run_name=run_name):
        # predictions
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # ROC
        fig1, ax1 = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax1)
        ax1.set_title("ROC Curve")
        mlflow.log_figure(fig1, "plots/roc_curve.png")
        plt.close(fig1)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig2, ax2 = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax2)
        ax2.set_title("Confusion Matrix")
        mlflow.log_figure(fig2, "plots/confusion_matrix.png")
        plt.close(fig2)

        # Feature importances
        importances = model.feature_importances_
        order = np.argsort(importances)
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.barh(np.array(feature_names)[order], importances[order])
        ax3.set_title("Feature Importances")
        ax3.set_xlabel("importance")
        mlflow.log_figure(fig3, "plots/feature_importances.png")
        plt.close(fig3)

    print("Plots logged to MLflow.")

if __name__ == "__main__":
    data = joblib.load("artifacts/gb_final_model.pkl")  # not correct usage; left for example
