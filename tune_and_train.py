# tune_and_train.py
import os
import time
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import uniform, randint

def setup_mlflow(local_dir="mlruns", experiment_name="Diabetes_Pipeline"):
    os.makedirs(local_dir, exist_ok=True)
    abs_uri = "file://" + os.path.abspath(local_dir)
    mlflow.set_tracking_uri(abs_uri)
    mlflow.set_experiment(experiment_name)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    return mlflow

def run_random_search(X_train, y_train,
                      param_distributions=None,
                      n_iter=40,
                      cv=5,
                      scoring="roc_auc",
                      random_state=42,
                      n_jobs=-1,
                      artifacts_dir="artifacts"):
    """
    Runs RandomizedSearchCV and logs results to MLflow. Returns the fitted RandomizedSearchCV object.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    if param_distributions is None:
        param_distributions = {
            "learning_rate": uniform(1e-3, 1.0),
            "max_depth": randint(1, 11),
            "n_estimators": randint(50, 401)
        }

    clf = GradientBoostingClassifier(random_state=random_state)

    rs = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=2,
        return_train_score=False
    )

    # Run and log
    with mlflow.start_run(run_name="RandomizedSearchCV"):
        rs.fit(X_train, y_train)
        best_params = rs.best_params_
        best_score = rs.best_score_
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_roc_auc", float(best_score))

        # Save cv results
        cv_df = pd.DataFrame(rs.cv_results_)
        cv_csv = os.path.join(artifacts_dir, "random_search_cv_results.csv")
        cv_df.to_csv(cv_csv, index=False)
        mlflow.log_artifact(cv_csv)

        # Log the best estimator as an MLflow model (with input example & signature)
        example_input = pd.DataFrame(X_train[:5], columns=[f"feature_{i}" for i in range(X_train.shape[1])])
        signature = infer_signature(X_train, rs.best_estimator_.predict(X_train))
        mlflow.sklearn.log_model(rs.best_estimator_, name="best_estimator", input_example=example_input, signature=signature)

    print("Random search complete. Best params:", best_params, "Best CV score (ROC-AUC):", best_score)
    return rs

def retrain_final_and_log(X_train_full, y_train_full, X_test, y_test, best_params, artifacts_dir="artifacts"):
    os.makedirs(artifacts_dir, exist_ok=True)
    model = GradientBoostingClassifier(random_state=42, **best_params)

    start = time.time()
    model.fit(X_train_full, y_train_full)
    train_time = time.time() - start

    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_prob_test)

    # Save locally
    model_path = os.path.join(artifacts_dir, "gb_final_model.pkl")
    joblib.dump(model, model_path)
    print(f"Final model saved to {model_path}")

    # Log to MLflow if run active
    if mlflow.active_run() is not None:
        example_input = pd.DataFrame(X_train_full[:5], columns=[f"feature_{i}" for i in range(X_train_full.shape[1])])
        signature = infer_signature(X_train_full, model.predict(X_train_full))
        mlflow.sklearn.log_model(model, name="final_model", input_example=example_input, signature=signature)

    return model, test_acc, test_auc, train_time, model_path

def register_model_mlflow(model_run_id, mlflow_client, model_name="Diabetes_GB_Model"):
    """
    Registers model in MLflow Model Registry using the run id.
    """
    model_uri = f"runs:/{model_run_id}/final_model"
    try:
        latest_registered = mlflow.register_model(model_uri=model_uri, name=model_name)
        print("Model registered:", latest_registered.name, latest_registered.version)
        return latest_registered
    except Exception as e:
        print("Model registry failed (common when using file-based mlruns):", e)
        return None

if __name__ == "__main__":
    # quick CLI usage: ensure artifacts/splits.npz exists
    splits = np.load("artifacts/splits.npz", allow_pickle=True)
    X_train = splits["X_train"]
    X_valid = splits["X_valid"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_valid = splits["y_valid"]
    y_test = splits["y_test"]

    setup_mlflow(local_dir="mlruns", experiment_name="Diabetes_Pipeline")

    # Run Randomized Search on the training set
    rs = run_random_search(X_train, y_train, n_iter=25)

    # Retrain on train+valid
    X_train_full = np.vstack([X_train, X_valid])
    y_train_full = np.concatenate([y_train, y_valid])
    best_params = rs.best_params_
    with mlflow.start_run(run_name="Final_Model"):
        model, test_acc, test_auc, train_time, model_path = retrain_final_and_log(
            X_train_full, y_train_full, X_test, y_test, best_params, artifacts_dir="artifacts"
        )
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "test_accuracy": float(test_acc),
            "test_roc_auc": float(test_auc),
            "train_time_s": float(train_time)
        })
        run_id = mlflow.active_run().info.run_id
        print("Final run id:", run_id)

    # Attempt register (may be unsupported on local file store)
    client = MlflowClient()
    register_model_mlflow(run_id, client, model_name="Diabetes_GB_Model")

    print("Done. Artifacts in artifacts/, mlruns/")
