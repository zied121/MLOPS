"""
model_pipeline.py  (Atelier 5 – MLflow integration)
----------------------------------------------------
Modularised ML pipeline for Customer Churn prediction.
Now includes MLflow tracking: params, metrics, artifacts, and model registry.

Functions
---------
prepare_data()   – Load and preprocess the dataset.
train_model()    – Train a RandomForest + log params to MLflow.
evaluate_model() – Evaluate + log metrics & confusion matrix to MLflow.
save_model()     – Persist model with joblib AND register with MLflow.
load_model()     – Reload a saved model from disk.
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend – safe in pipelines
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
EXPERIMENT_NAME     = "churn-prediction"



# ---------------------------------------------------------------------------
# 1. prepare_data
# ---------------------------------------------------------------------------

def prepare_data(filepath: str, test_size: float = 0.2, random_state: int = 1):
    """
    Load and preprocess the Churn Modelling CSV dataset.

    Steps
    -----
    - Read CSV
    - Drop irrelevant columns (RowNumber, CustomerId, Surname, Geography)
    - Encode the Gender column with LabelEncoder
    - Split features / target
    - Train / test split

    Parameters
    ----------
    filepath     : path to Churn_Modelling.csv
    test_size    : fraction of data reserved for testing  (default 0.2)
    random_state : random seed for reproducibility        (default 1)

    Returns
    -------
    x_train, x_test, y_train, y_test
    """
    df = pd.read_csv(filepath)
    print(f"[prepare_data] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    missing    = df.isna().sum().sum()
    duplicates = df.duplicated().sum()
    print(f"[prepare_data] Missing values: {missing} | Duplicates: {duplicates}")

    encoder        = LabelEncoder()
    df["Gender"]   = encoder.fit_transform(df["Gender"])

    columns_to_drop = ["RowNumber", "CustomerId", "Surname", "Geography"]
    df = df.drop(columns=columns_to_drop)

    x = df.drop(columns=["Exited"])
    y = df["Exited"]

    print(f"[prepare_data] Features: {list(x.columns)}")
    print(f"[prepare_data] Target distribution:\n{y.value_counts().to_string()}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    print(f"[prepare_data] Train size: {len(x_train)} | Test size: {len(x_test)}")

    return x_train, x_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. train_model  – logs hyperparams to MLflow
# ---------------------------------------------------------------------------

def train_model(
    x_train,
    y_train,
    n_estimators: int = 100,
    random_state: int = 42,
    max_depth=None,
):
    """
    Train a RandomForestClassifier and log hyperparameters to MLflow.

    Parameters
    ----------
    x_train      : training features
    y_train      : training labels
    n_estimators : number of trees  (default 100)
    random_state : random seed      (default 42)
    max_depth    : max tree depth   (default None = unlimited)

    Returns
    -------
    model : fitted RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
    )
    model.fit(x_train, y_train)
    print(f"[train_model] RandomForest trained with {n_estimators} estimators.")

    # ── MLflow: log hyperparameters ────────────────────────────────────────
    mlflow.log_param("n_estimators",  n_estimators)
    mlflow.log_param("random_state",  random_state)
    mlflow.log_param("max_depth",     str(max_depth))
    mlflow.log_param("model_type",    "RandomForestClassifier")
    mlflow.log_param("train_samples", len(x_train))

    return model


# ---------------------------------------------------------------------------
# 3. evaluate_model  – logs metrics & confusion matrix artifact to MLflow
# ---------------------------------------------------------------------------

def evaluate_model(model, x_test, y_test, show_plot: bool = False):
    """
    Evaluate the trained model and log metrics + confusion matrix to MLflow.

    Parameters
    ----------
    model      : fitted sklearn estimator
    x_test     : test features
    y_test     : true test labels
    show_plot  : display plot interactively (default False – use Agg backend)

    Returns
    -------
    metrics : dict with accuracy, precision, recall, f1, report, confusion_matrix
    """
    y_pred = model.predict(x_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    report    = classification_report(
        y_test, y_pred, target_names=["Not Exited", "Exited"]
    )
    matrix    = confusion_matrix(y_test, y_pred)

    print(f"[evaluate_model] Accuracy  : {accuracy * 100:.2f}%")
    print(f"[evaluate_model] Precision : {precision:.4f}")
    print(f"[evaluate_model] Recall    : {recall:.4f}")
    print(f"[evaluate_model] F1-Score  : {f1:.4f}")
    print(f"[evaluate_model] Classification Report:\n{report}")

    # ── MLflow: log metrics ────────────────────────────────────────────────
    mlflow.log_metric("accuracy",     round(accuracy,  4))
    mlflow.log_metric("precision",    round(precision, 4))
    mlflow.log_metric("recall",       round(recall,    4))
    mlflow.log_metric("f1_score",     round(f1,        4))
    mlflow.log_metric("test_samples", len(x_test))

    # ── MLflow: save & log confusion matrix as artifact ───────────────────
    cm_path = "confusion_matrix.png"
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=["Not Exited", "Exited"]
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix – Churn Prediction")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=100)
    plt.close()
    mlflow.log_artifact(cm_path)
    print(f"[evaluate_model] Confusion matrix saved & logged → {cm_path}")

    return {
        "accuracy":         accuracy,
        "precision":        precision,
        "recall":           recall,
        "f1_score":         f1,
        "report":           report,
        "confusion_matrix": matrix,
    }


# ---------------------------------------------------------------------------
# 4. save_model  – saves with joblib AND registers with MLflow
# ---------------------------------------------------------------------------

def save_model(model, filepath: str = "classifier.joblib"):
    """
    Save the model with joblib and log it to MLflow model registry.

    Parameters
    ----------
    model    : fitted sklearn estimator
    filepath : local joblib path (default 'classifier.joblib')
    """
    # joblib save (for FastAPI / direct loading)
    joblib.dump(model, filepath)
    print(f"[save_model] Model saved → {filepath}")

    # ── MLflow: log model to registry ─────────────────────────────────────
    mlflow.sklearn.log_model(
        sk_model        = model,
        artifact_path   = "random_forest_model",
        registered_model_name = "ChurnPredictor",
    )
    mlflow.log_artifact(filepath)
    print("[save_model] Model registered in MLflow as 'ChurnPredictor'")


# ---------------------------------------------------------------------------
# 5. load_model
# ---------------------------------------------------------------------------

def load_model(filepath: str = "classifier.joblib"):
    """
    Load a previously saved model from disk.

    Parameters
    ----------
    filepath : path to the joblib file (default 'classifier.joblib')

    Returns
    -------
    model : loaded sklearn estimator
    """
    model = joblib.load(filepath)
    print(f"[load_model] Model loaded ← {filepath}")
    return model


# ---------------------------------------------------------------------------
# 6. run_experiment  – convenience wrapper: one full MLflow run
# ---------------------------------------------------------------------------

def run_experiment(
    filepath: str   = "Churn_Modelling.csv",
    n_estimators: int = 100,
    max_depth       = None,
    test_size: float = 0.2,
    model_path: str  = "classifier.joblib",
):
    """
    Run a complete experiment tracked by MLflow:
    prepare → train → evaluate → save — all inside one mlflow.start_run().

    Parameters
    ----------
    filepath     : path to CSV
    n_estimators : RF hyperparameter
    max_depth    : RF hyperparameter
    test_size    : train/test split ratio
    model_path   : where to save the joblib model

    Returns
    -------
    metrics : dict returned by evaluate_model()
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        # Log dataset info
        mlflow.log_param("dataset",   filepath)
        mlflow.log_param("test_size", test_size)

        x_train, x_test, y_train, y_test = prepare_data(
            filepath, test_size=test_size
        )
        model   = train_model(x_train, y_train, n_estimators=n_estimators,
                               max_depth=max_depth)
        metrics = evaluate_model(model, x_test, y_test)
        save_model(model, model_path)

        print(f"\n✅ MLflow run complete — experiment: '{EXPERIMENT_NAME}'")
        print(f"   Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"   UI: http://127.0.0.1:5000")

    return metrics
