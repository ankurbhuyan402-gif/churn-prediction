import os
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_DIR = BASE_DIR / "models"

MLFLOW_DB = BASE_DIR / "mlflow.db"
MLFLOW_ARTIFACTS = BASE_DIR / "mlflow_artifacts"


# Create required directories
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. MLflow CONFIGURATION
# =========================================================

# SQLite database for MLflow tracking
mlflow.set_tracking_uri(
    "sqlite:///" + MLFLOW_DB.as_posix()
)

# New experiment name
EXPERIMENT_NAME = "Telco_Churn_Prediction_Lab4"


# Create experiment if it does not already exist
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:

    mlflow.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location=MLFLOW_ARTIFACTS.as_uri()
    )


# Select experiment
mlflow.set_experiment(EXPERIMENT_NAME)


# =========================================================
# 3. TRAIN AND TRACK FUNCTION
# =========================================================

def train_and_track(run_name="RandomForest_Baseline", params=None):

    # Default Random Forest parameters
    if params is None:

        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced"
        }

    print(f"\n--- Starting MLflow Run: {run_name} ---")


    # =====================================================
    # 4. LOAD PROCESSED DATA
    # =====================================================

    X_train = np.load(
        DATA_DIR / "X_train_final.npy"
    )

    X_test = np.load(
        DATA_DIR / "X_test_final.npy"
    )

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )


    print("Training data shape:", X_train.shape)
    print("Testing data shape :", X_test.shape)


    # =====================================================
    # 5. START MLFLOW RUN
    # =====================================================

    with mlflow.start_run(run_name=run_name):


        # =================================================
        # 6. LOG PARAMETERS
        # =================================================

        mlflow.log_params(params)

        mlflow.log_param(
            "model_family",
            "RandomForest"
        )


        # =================================================
        # 7. CREATE MODEL
        # =================================================

        model = RandomForestClassifier(
            **params
        )


        # =================================================
        # 8. TRAIN MODEL
        # =================================================

        model.fit(
            X_train,
            y_train
        )


        # =================================================
        # 9. MAKE PREDICTIONS
        # =================================================

        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]


        # =================================================
        # 10. CALCULATE METRICS
        # =================================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )


        # =================================================
        # 11. LOG METRICS
        # =================================================

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        mlflow.log_metrics(
            metrics
        )


        print(
            f"Metrics logged: "
            f"F1 = {f1:.4f} | "
            f"ROC-AUC = {roc_auc:.4f}"
        )


        # =================================================
        # 12. CONFUSION MATRIX
        # =================================================

        fig_cm, ax_cm = plt.subplots(
            figsize=(6, 5)
        )

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax_cm,
            cmap="Blues"
        )

        ax_cm.set_title(
            f"Confusion Matrix - {run_name}"
        )

        cm_path = ARTIFACT_DIR / "confusion_matrix.png"

        fig_cm.savefig(
            cm_path,
            bbox_inches="tight"
        )

        plt.close(fig_cm)


        # Log confusion matrix to MLflow
        mlflow.log_artifact(
            str(cm_path),
            artifact_path="plots"
        )


        # =================================================
        # 13. ROC CURVE
        # =================================================

        fig_roc, ax_roc = plt.subplots(
            figsize=(6, 5)
        )

        RocCurveDisplay.from_predictions(
            y_test,
            y_prob,
            ax=ax_roc
        )

        ax_roc.set_title(
            f"ROC Curve - {run_name}"
        )

        roc_path = ARTIFACT_DIR / "roc_curve.png"

        fig_roc.savefig(
            roc_path,
            bbox_inches="tight"
        )

        plt.close(fig_roc)


        # Log ROC curve to MLflow
        mlflow.log_artifact(
            str(roc_path),
            artifact_path="plots"
        )


        # =================================================
        # 14. LOG DATASET METADATA
        # =================================================

        metadata_path = (
            DATA_DIR / "dataset_metadata.json"
        )

        if metadata_path.exists():

            mlflow.log_artifact(
                str(metadata_path),
                artifact_path="metadata"
            )


        # =================================================
        # 15. LOG MODEL TO MLFLOW
        # =================================================

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )


        # =================================================
        # 16. SAVE LOCAL MODEL BACKUP
        # =================================================

        model_path = (
            MODEL_DIR / "random_forest_model.pkl"
        )

        joblib.dump(
            model,
            model_path
        )


        # =================================================
        # 17. FINISHED
        # =================================================

        print(
            f"Run '{run_name}' successfully tracked!"
        )


# =========================================================
# 18. MAIN
# =========================================================

if __name__ == "__main__":

    # Lab 4 baseline Random Forest
    train_and_track(
        run_name="RandomForest"
    )
    print("\nAll runs completed and tracked in MLflow.")