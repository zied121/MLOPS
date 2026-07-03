"""
pipeline_prefect.py  (Atelier 3 + 4)
--------------------------------------
Prefect pipeline for the Customer Churn ML project.

Flows
-----
  all        – clone + install + code quality + data + train + save + evaluate
  train      – prepare data + train + save
  evaluate   – load + evaluate + predict
  code       – format + lint + security + unit tests
  api        – load model + start FastAPI server
  setup      – clone/pull repo only

Usage
-----
  python pipeline_prefect.py --flow all
  python pipeline_prefect.py --flow train
  python pipeline_prefect.py --flow evaluate
  python pipeline_prefect.py --flow code
  python pipeline_prefect.py --flow api
  python pipeline_prefect.py --flow setup
"""

import argparse
import os
import subprocess
import sys

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

DATA_PATH     = "Churn_Modelling.csv"
MODEL_PATH    = "classifier.joblib"
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
REPO_URL      = "https://github.com/zied121/MLOPS.git"   # ← change this


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


# ============================================================================
# TASKS – Git / Repo  (Étape 06 – Atelier 3)
# ============================================================================

@task(name="clone-or-pull-repo")
def task_clone_repo(repo_url: str = REPO_URL):
    """
    Clone the project repo if not present locally,
    or pull the latest changes if it already exists.
    """
    logger = get_run_logger()

    project_dir = os.path.basename(repo_url).replace(".git", "")

    if os.path.exists(os.path.join(project_dir, ".git")):
        logger.info(f"📥 Repo '{project_dir}' already cloned — pulling latest changes ...")
        result = _run(["git", "-C", project_dir, "pull"])
    else:
        logger.info(f"📦 Cloning repo: {repo_url}")
        result = _run(["git", "clone", repo_url])

    if result.returncode == 0:
        logger.info(f"✅ Repo up to date: {project_dir}")
    else:
        logger.warning(f"⚠️  Git error:\n{result.stderr}")

    return result.returncode


# ============================================================================
# TASKS – Infrastructure
# ============================================================================

@task(name="install-dependencies")
def task_install_dependencies():
    logger = get_run_logger()
    logger.info("📦 Installing dependencies from requirements.txt ...")
    result = _run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    if result.returncode == 0:
        logger.info("✅ Dependencies installed successfully.")
    else:
        logger.warning(f"⚠️  pip output:\n{result.stderr}")
    return result.returncode


# ============================================================================
# TASKS – Code quality
# ============================================================================

@task(name="format-code")
def task_format_code():
    logger = get_run_logger()
    logger.info(f"🎨 Formatting: {PROJECT_FILES}")
    _run([sys.executable, "-m", "pip", "install", "black", "-q"])
    result = _run([sys.executable, "-m", "black"] + PROJECT_FILES)
    logger.info(result.stdout or "Black: no output")
    if result.returncode == 0:
        logger.info("✅ Code formatted.")
    else:
        logger.warning(result.stderr)
    return result.returncode


@task(name="code-quality")
def task_code_quality():
    logger = get_run_logger()
    logger.info(f"🔍 Linting: {PROJECT_FILES}")
    _run([sys.executable, "-m", "pip", "install", "flake8", "-q"])
    result = _run(
        [sys.executable, "-m", "flake8", "--max-line-length=100",
         "--exclude=venv,.venv,__pycache__"] + PROJECT_FILES
    )
    if result.returncode == 0:
        logger.info("✅ No linting issues found.")
    else:
        logger.warning(f"⚠️  Flake8 issues:\n{result.stdout}")
    return result.returncode


@task(name="security-check")
def task_security_check():
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
# TASKS – ML pipeline
# ============================================================================

@task(name="prepare-data")
def task_prepare_data():
    logger = get_run_logger()
    logger.info(f"📂 Preparing data from {DATA_PATH} ...")
    x_train, x_test, y_train, y_test = prepare_data(DATA_PATH)
    logger.info(f"✅ Data ready — train: {len(x_train)}, test: {len(x_test)}")
    return x_train, x_test, y_train, y_test


@task(name="train-model")
def task_train_model(x_train, y_train):
    logger = get_run_logger()
    logger.info("🤖 Training model ...")
    model = train_model(x_train, y_train)
    logger.info("✅ Model trained.")
    return model


@task(name="save-model")
def task_save_model(model):
    logger = get_run_logger()
    logger.info(f"💾 Saving model → {MODEL_PATH}")
    save_model(model, MODEL_PATH)
    logger.info("✅ Model saved.")


@task(name="load-model")
def task_load_model():
    logger = get_run_logger()
    logger.info(f"📥 Loading model ← {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    logger.info("✅ Model loaded.")
    return model


@task(name="evaluate-model")
def task_evaluate_model(model, x_test, y_test):
    logger = get_run_logger()
    logger.info("📊 Evaluating model ...")
    metrics = evaluate_model(model, x_test, y_test, show_plot=False)
    logger.info(f"✅ Accuracy: {metrics['accuracy'] * 100:.2f}%")
    logger.info(f"Classification Report:\n{metrics['report']}")
    return metrics


@task(name="predict")
def task_predict(model):
    logger = get_run_logger()
    logger.info("🔮 Running sample prediction ...")
    sample = [[850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]]
    prediction = model.predict(sample)
    result = "Exited" if prediction[0] == 1 else "Not Exited"
    logger.info(f"✅ Prediction for sample input: {result}")
    return result


# ============================================================================
# TASK – API  (Atelier 4)
# ============================================================================

@task(name="start-api")
def task_start_api(host: str = "0.0.0.0", port: int = 8000):
    """Start FastAPI server with Uvicorn in background. Logs → uvicorn.log"""
    logger = get_run_logger()
    logger.info(f"🌐 Starting FastAPI on http://{host}:{port} ...")
    _run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "-q"])
    cmd = [
        sys.executable, "-m", "uvicorn", "app:app",
        "--host", host, "--port", str(port), "--reload",
    ]
    log_file = open("uvicorn.log", "w")
    subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
    logger.info(f"✅ API running  → http://{host}:{port}")
    logger.info(f"📖 Swagger docs → http://127.0.0.1:{port}/docs")
    logger.info("📄 Logs → uvicorn.log")
    return f"http://{host}:{port}"


# ============================================================================
# FLOWS
# ============================================================================

@flow(name="setup", log_prints=True)
def flow_setup():
    """
    Setup flow:
    clone/pull repo only
    """
    task_clone_repo()


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
    prepare data → train → save
    """
    x_train, x_test, y_train, y_test = task_prepare_data()
    model = task_train_model(x_train, y_train)
    task_save_model(model)


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    """
    Evaluation flow:
    load → prepare data → evaluate → predict
    """
    model = task_load_model()
    x_train, x_test, y_train, y_test = task_prepare_data()
    task_evaluate_model(model, x_test, y_test)
    task_predict(model)


@flow(name="api", log_prints=True)
def flow_api():
    """
    API flow:
    load model → start FastAPI server
    """
    task_load_model()
    task_start_api()


@flow(name="all", log_prints=True)
def flow_all():
    """
    Full pipeline:
    clone/pull → install → code quality → data → train → save → evaluate
    """
    # Step 0 – Git: clone or pull latest code
    task_clone_repo()

    # Step 1 – Install dependencies
    task_install_dependencies()

    # Step 2 – Code quality
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
    "all":      flow_all,
    "setup":    flow_setup,
    "train":    flow_train,
    "evaluate": flow_evaluate,
    "code":     flow_code,
    "api":      flow_api,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefect ML Pipeline – Atelier 4")
    parser.add_argument(
        "--flow",
        type=str,
        choices=list(FLOWS.keys()),
        default="all",
        help="Flow to run: all | setup | train | evaluate | code | api  (default: all)",
    )
    args = parser.parse_args()
    print(f"\n🚀 Starting flow: '{args.flow}'\n")
    FLOWS[args.flow]()
