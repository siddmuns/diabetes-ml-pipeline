# visualize_and_evaluate.py
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

def make_plots_and_log(model, X_test, y_test, feature_names, run_name="Final_Plots"):
    """
    Generate ROC curve, confusion matrix, feature importance plots and log them to MLflow.
    Handles nested runs if another run is already active.
    """
    active_run = mlflow.active_run()

    # If a run is already active, nest this one
    if active_run:
        run_context = mlflow.start_run(run_name=run_name, nested=True)
    else:
        run_context = mlflow.start_run(run_name=run_name)

    with run_context:
        # ROC Curve
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig("artifacts/roc_curve.png")
        mlflow.log_artifact("artifacts/roc_curve.png")

        # Confusion Matrix
        cm = confusion_matrix(y_test, model.predict(X_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.savefig("artifacts/confusion_matrix.png")
        mlflow.log_artifact("artifacts/confusion_matrix.png")

        # Feature Importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            plt.figure(figsize=(8, 6))
            plt.barh(feature_names, importances)
            plt.xlabel("Importance")
            plt.title("Feature Importances")
            plt.tight_layout()
            plt.savefig("artifacts/feature_importances.png")
            mlflow.log_artifact("artifacts/feature_importances.png")

        mlflow.log_metric("roc_auc", float(roc_auc))

    print(f"Visualization complete. ROC-AUC: {roc_auc:.3f}")
