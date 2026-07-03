"""
model_pipeline.py
-----------------
Modularised ML pipeline for Customer Churn prediction.
Extracted from customer_churn.ipynb (Atelier 2 – Modularisation du Code).

Functions
---------
prepare_data()   – Load and preprocess the dataset.
train_model()    – Train a RandomForest classifier.
evaluate_model() – Evaluate accuracy + confusion matrix.
save_model()     – Persist the trained model with joblib.
load_model()     – Reload a saved model from disk.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

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
    x_train, x_test, y_train, y_test : numpy arrays / DataFrames
    """
    # --- Load ---
    df = pd.read_csv(filepath)
    print(f"[prepare_data] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # --- Basic checks ---
    missing = df.isna().sum().sum()
    duplicates = df.duplicated().sum()
    print(f"[prepare_data] Missing values: {missing} | Duplicates: {duplicates}")

    # --- Encode categorical ---
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # --- Drop non-predictive columns ---
    columns_to_drop = ["RowNumber", "CustomerId", "Surname", "Geography"]
    df = df.drop(columns=columns_to_drop)

    # --- Features / target split ---
    x = df.drop(columns=["Exited"])
    y = df["Exited"]

    print(f"[prepare_data] Features: {list(x.columns)}")
    print(f"[prepare_data] Target distribution:\n{y.value_counts().to_string()}")

    # --- Train / test split ---
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    print(f"[prepare_data] Train size: {len(x_train)} | Test size: {len(x_test)}")

    return x_train, x_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. train_model
# ---------------------------------------------------------------------------


def train_model(x_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """
    Train a RandomForestClassifier on the provided training data.

    Parameters
    ----------
    x_train      : training features
    y_train      : training labels
    n_estimators : number of trees in the forest (default 100)
    random_state : random seed                    (default 42)

    Returns
    -------
    model : fitted RandomForestClassifier
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    print(f"[train_model] RandomForest trained with {n_estimators} estimators.")
    return model


# ---------------------------------------------------------------------------
# 3. evaluate_model
# ---------------------------------------------------------------------------


def evaluate_model(model, x_test, y_test, show_plot: bool = True):
    """
    Evaluate the trained model on the test set.

    Prints accuracy and a full classification report.
    Optionally displays the confusion matrix.

    Parameters
    ----------
    model      : fitted sklearn estimator
    x_test     : test features
    y_test     : true test labels
    show_plot  : whether to display the confusion matrix plot (default True)

    Returns
    -------
    metrics : dict with keys 'accuracy', 'report', 'confusion_matrix'
    """
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Not Exited", "Exited"]
    )
    matrix = confusion_matrix(y_test, y_pred)

    print(f"[evaluate_model] Accuracy : {accuracy * 100:.2f}%")
    print(f"[evaluate_model] Classification Report:\n{report}")

    if show_plot:
        disp = ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=["Not Exited", "Exited"]
        )
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix – Churn Prediction")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=100)
        print("[evaluate_model] Confusion matrix saved → confusion_matrix.png")
        plt.show()

    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": matrix,
    }


# ---------------------------------------------------------------------------
# 4. save_model
# ---------------------------------------------------------------------------


def save_model(model, filepath: str = "classifier.joblib"):
    """
    Serialise and save the trained model to disk using joblib.

    Parameters
    ----------
    model    : fitted sklearn estimator
    filepath : destination file path (default 'classifier.joblib')
    """
    joblib.dump(model, filepath)
    print(f"[save_model] Model saved → {filepath}")


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
