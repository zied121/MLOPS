"""
-------------------
Prefect pipeline for the Customer Churn ML project.
Orchestrates: install → code quality → data → train → evaluate → predict.

Flows available
---------------
  all        – Full pipeline (install + code + data + train + save + evaluate)
  train      – Prepare data + train + save model
  evaluate   – Load model + evaluate
  code       – Format + lint + security + unit tests only

Usage
-----
  python pipeline_prefect.py --flow all
  python pipeline_prefect.py --flow train
  python pipeline_prefect.py --flow evaluate
  python pipeline_prefect.py --flow code
"""

import argparse
import subprocess
import sys
import os

from prefect import flow, task, get_run_logger

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"

# Only scan these project files for quality / security checks
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command and return its CompletedProcess."""
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


# ============================================================================
# TASKS – Section 1 : Infrastructure
# ============================================================================


@task(name="install-dependencies")
def task_install_dependencies():
    """Install all packages listed in requirements.txt."""
    logger = get_run_logger()
    logger.info("📦 Installing dependencies from requirements.txt ...")
    result = _run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    if result.returncode == 0:
        logger.info("✅ Dependencies installed successfully.")
    else:
        logger.warning(f"⚠️  pip output:\n{result.stderr}")
    return result.returncode


# ============================================================================
# TASKS – Section 2 : Code quality
# ============================================================================


@task(name="format-code")
def task_format_code():
    """
    Auto-format project files with Black.
    Scans only: model_pipeline.py, main.py, pipeline_prefect.py
    """
    logger = get_run_logger()
    logger.info(f"🎨 Formatting: {PROJECT_FILES}")

    # Install black if missing
    _run([sys.executable, "-m", "pip", "install", "black", "-q"])

    result = _run([sys.executable, "-m", "black"] + PROJECT_FILES)
    logger.info(result.stdout or "Black: no output")
    if result.returncode != 0:
        logger.warning(result.stderr)
    else:
        logger.info("✅ Code formatted.")
    return result.returncode


@task(name="code-quality")
def task_code_quality():
    """
    Run Flake8 linter on project files only.
    Scans only: model_pipeline.py, main.py, pipeline_prefect.py
    """
    logger = get_run_logger()
    logger.info(f"🔍 Linting: {PROJECT_FILES}")

    _run([sys.executable, "-m", "pip", "install", "flake8", "-q"])

    result = _run(
        [
            sys.executable,
            "-m",
            "flake8",
            "--max-line-length=100",
            "--exclude=venv,.venv,__pycache__",
        ]
        + PROJECT_FILES
    )
    if result.returncode == 0:
        logger.info("✅ No linting issues found.")
    else:
        logger.warning(f"⚠️  Flake8 issues:\n{result.stdout}")
    return result.returncode


@task(name="security-check")
def task_security_check():
    """
    Run Bandit security scanner on project files only.
    Scans only: model_pipeline.py, main.py, pipeline_prefect.py
    """
    logger = get_run_logger()
    logger.info(f"🔒 Security scan: {PROJECT_FILES}")

    _run([sys.executable, "-m", "pip", "install", "bandit", "-q"])

    result = _run([sys.executable, "-m", "bandit", "-ll", "-q"] + PROJECT_FILES)
    if result.returncode == 0:
        logger.info("✅ No security issues found.")
    else:
        logger.warning(f"⚠️  Bandit findings:\n{result.stdout}\n{result.stderr}")
    return result.returncode


@task(name="run-unit-tests")
def task_run_unit_tests():
    """
    Install pytest and run unit tests from test_pipeline.py.
    """
    logger = get_run_logger()
    logger.info("🧪 Running unit tests ...")

    _run([sys.executable, "-m", "pip", "install", "pytest", "-q"])

    result = _run([sys.executable, "-m", "pytest", "test_pipeline.py", "-v"])
    logger.info(result.stdout)
    if result.returncode == 0:
        logger.info("✅ All tests passed.")
    else:
        logger.error(f"❌ Some tests failed:\n{result.stderr}")
    return result.returncode


# ============================================================================
# TASKS – Section 3 : ML pipeline
# ============================================================================


@task(name="prepare-data")
def task_prepare_data():
    """Load and preprocess the Churn dataset."""
    logger = get_run_logger()
    logger.info(f"📂 Preparing data from {DATA_PATH} ...")
    x_train, x_test, y_train, y_test = prepare_data(DATA_PATH)
    logger.info(f"✅ Data ready — train: {len(x_train)}, test: {len(x_test)}")
    return x_train, x_test, y_train, y_test


@task(name="train-model")
def task_train_model(x_train, y_train):
    """Train a RandomForest classifier."""
    logger = get_run_logger()
    logger.info("🤖 Training model ...")
    model = train_model(x_train, y_train)
    logger.info("✅ Model trained.")
    return model


@task(name="save-model")
def task_save_model(model):
    """Persist the trained model to disk."""
    logger = get_run_logger()
    logger.info(f"💾 Saving model → {MODEL_PATH}")
    save_model(model, MODEL_PATH)
    logger.info("✅ Model saved.")


@task(name="load-model")
def task_load_model():
    """Load the saved model from disk."""
    logger = get_run_logger()
    logger.info(f"📥 Loading model ← {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    logger.info("✅ Model loaded.")
    return model


@task(name="evaluate-model")
def task_evaluate_model(model, x_test, y_test):
    """Evaluate the model and log metrics."""
    logger = get_run_logger()
    logger.info("📊 Evaluating model ...")
    metrics = evaluate_model(model, x_test, y_test, show_plot=False)
    logger.info(f"✅ Accuracy: {metrics['accuracy'] * 100:.2f}%")
    logger.info(f"Classification Report:\n{metrics['report']}")
    return metrics


@task(name="predict")
def task_predict(model):
    """Run a sample prediction with the loaded model."""
    logger = get_run_logger()
    logger.info("🔮 Running sample prediction ...")
    # Sample: CreditScore, Gender, Age, Tenure, Balance,
    #         NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
    sample = [[850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]]
    prediction = model.predict(sample)
    result = "Exited" if prediction[0] == 1 else "Not Exited"
    logger.info(f"✅ Prediction for sample input: {result}")
    return result


# ============================================================================
# FLOWS
# ============================================================================


@flow(name="code", log_prints=True)
def flow_code():
    """
    Code quality flow:
    format → lint → security → unit tests
    """
    task_format_code()
    task_code_quality()
    task_security_check()
    task_run_unit_tests()


@flow(name="train", log_prints=True)
def flow_train():
    """
    Training flow:
    prepare data → train model → save model
    """
    x_train, x_test, y_train, y_test = task_prepare_data()
    model = task_train_model(x_train, y_train)
    task_save_model(model)


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    """
    Evaluation flow:
    load model → prepare data → evaluate → predict
    """
    model = task_load_model()
    x_train, x_test, y_train, y_test = task_prepare_data()
    task_evaluate_model(model, x_test, y_test)
    task_predict(model)


@flow(name="all", log_prints=True)
def flow_all():
    """
    Full pipeline:
    install → code quality → prepare data → train → save → evaluate
    """
    # Step 1 – Install dependencies
    task_install_dependencies()

    # Step 2 – Code quality (format, lint, security, tests)
    task_format_code()
    task_code_quality()
    task_security_check()
    task_run_unit_tests()

    # Step 3 – ML pipeline
    x_train, x_test, y_train, y_test = task_prepare_data()
    model = task_train_model(x_train, y_train)
    task_save_model(model)
    task_evaluate_model(model, x_test, y_test)
    task_predict(model)


# ============================================================================
# Entry point – CLI
# ============================================================================

FLOWS = {
    "all": flow_all,
    "train": flow_train,
    "evaluate": flow_evaluate,
    "code": flow_code,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefect ML Pipeline – Atelier 3")
    parser.add_argument(
        "--flow",
        type=str,
        choices=list(FLOWS.keys()),
        default="all",
        help="Flow to run: all | train | evaluate | code  (default: all)",
    )
    args = parser.parse_args()

    selected_flow = FLOWS[args.flow]
    print(f"\n🚀 Starting flow: '{args.flow}'\n")
    selected_flow()
