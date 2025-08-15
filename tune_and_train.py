# tune_and_train.py
import os
import time
import joblib
import json
import numpy as np
import optuna
import mlflow
import mlflow.sklearn

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from mlflow import MlflowClient

def setup_mlflow(local_dir="mlruns", experiment_name="Diabetes_Pipeline"):
    os.makedirs(local_dir, exist_ok=True)
    abs_uri = "file://" + os.path.abspath(local_dir)
    mlflow.set_tracking_uri(abs_uri)
    mlflow.set_experiment(experiment_name)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    return mlflow

def objective_factory(X_train, y_train, X_valid, y_valid):
    """
    Returns an objective function that captures training/validation data closures.
    The function will be used by Optuna and will create nested MLflow runs.
    """
    def objective(trial):
        # suggest hyperparams
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1.0, log=True)
        max_depth = trial.suggest_int("max_depth", 1, 10)
        n_estimators = trial.suggest_int("n_estimators", 50, 400)

        # Start an MLflow nested run to log this trial
        with mlflow.start_run(nested=True):
            params = {"learning_rate": learning_rate, "max_depth": max_depth, "n_estimators": n_estimators}
            mlflow.log_params(params)

            model = GradientBoostingClassifier(
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=n_estimators,
                random_state=42
            )
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            # evaluate on validation
            y_prob = model.predict_proba(X_valid)[:, 1]
            y_pred = model.predict(X_valid)
            acc = accuracy_score(y_valid, y_pred)
            auc = roc_auc_score(y_valid, y_prob)

            mlflow.log_metrics({"val_accuracy": acc, "val_roc_auc": auc, "train_time_s": train_time})

            # save model artifact for this trial (optional)
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Optuna minimizes -> return negative AUC so it effectively maximizes AUC
            return -auc
    return objective

def run_optuna(X_train, y_train, X_valid, y_valid, n_trials=25, study_name="diabetes_study"):
    study = optuna.create_study(direction="minimize", study_name=study_name)
    objective = objective_factory(X_train, y_train, X_valid, y_valid)
    study.optimize(objective, n_trials=n_trials)
    print("Optuna best trial:", study.best_trial.params, "best value:", -study.best_value)
    return study

def retrain_final_and_log(X_train_full, y_train_full, X_test, y_test, best_params, artifacts_dir="artifacts"):
    os.makedirs(artifacts_dir, exist_ok=True)
    # Retrain on train + valid (X_train_full already should be train+valid)
    model = GradientBoostingClassifier(random_state=42, **best_params)
    import time
    start = time.time()
    model.fit(X_train_full, y_train_full)
    train_time = time.time() - start

    # Evaluate on test
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_prob_test)

    # Save artifacts locally
    model_path = os.path.join(artifacts_dir, "gb_final_model.pkl")
    joblib.dump(model, model_path)

    print(f"Final model saved to {model_path}")
    return model, test_acc, test_auc, train_time, model_path

def register_model_mlflow(model_run_id, mlflow_client, model_name="Diabetes_GB_Model"):
    """
    Attempts to register the model into MLflow Model Registry using the run id.
    model_run_id: run id where the final model was logged as artifact at path 'final_model' or 'model'
    """
    model_uri = f"runs:/{model_run_id}/final_model"
    try:
        # If the artifact path is 'model' change accordingly:
        # fallback to 'model' if 'final_model' does not exist
        latest_registered = mlflow.register_model(model_uri=model_uri, name=model_name)
        print("Model registered:", latest_registered.name, latest_registered.version)
        return latest_registered
    except Exception as e:
        print("Model registry failed:", e)
        return None

if __name__ == "__main__":
    # Example CLI flow: assume splits.npz created by data_ingestion
    import numpy as np
    splits = np.load("artifacts/splits.npz", allow_pickle=True)
    X_train = splits["X_train"]
    X_valid = splits["X_valid"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_valid = splits["y_valid"]
    y_test = splits["y_test"]
    feature_names = list(splits["feature_names"])

    # setup mlflow
    setup_mlflow(local_dir="mlruns", experiment_name="Diabetes_Pipeline")

    # run optuna
    study = run_optuna(X_train, y_train, X_valid, y_valid, n_trials=20)

    best_params = study.best_trial.params
    print("Best params (Optuna):", best_params)

    # retrain final model on train+valid
    X_train_full = np.vstack([X_train, X_valid])
    y_train_full = np.concatenate([y_train, y_valid])

    # Start a final MLflow run to log final model and metrics
    with mlflow.start_run(run_name="Final_Model") as final_run:
        model, test_acc, test_auc, train_time, model_path = retrain_final_and_log(
            X_train_full, y_train_full, X_test, y_test, best_params, artifacts_dir="artifacts"
        )
        # Log params and metrics
        mlflow.log_params(best_params)
        mlflow.log_metrics({"test_accuracy": float(test_acc), "test_roc_auc": float(test_auc), "train_time_s": float(train_time)})
        # Log model artifact to MLflow under artifact path final_model
        mlflow.sklearn.log_model(model, artifact_path="final_model")
        run_id = final_run.info.run_id
        print("Final run id:", run_id)

    # Attempt to register model in MLflow model registry
    client = MlflowClient()
    register_model_mlflow(run_id, client, model_name="Diabetes_GB_Model")

    # Save scaler.pkl (already saved by data_ingestion)
    print("Done. Artifacts in artifacts/, mlruns/")
