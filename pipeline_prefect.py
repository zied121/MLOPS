"""
pipeline_prefect.py  (Atelier 3 → 6)
--------------------------------------
Prefect pipeline for the Customer Churn ML project.

Flows
-----
  all        – clone + install + code quality + data + train + save + evaluate
  setup      – clone/pull repo only
  train      – prepare data + train + save
  evaluate   – load + evaluate + predict
  code       – format + lint + security + unit tests
  api        – load model + start FastAPI server
  mlflow     – run MLflow experiment + start UI
  cd         – docker build + run + tag + push to Docker Hub

Usage
-----
  python pipeline_prefect.py --flow all
  python pipeline_prefect.py --flow train
  python pipeline_prefect.py --flow evaluate
  python pipeline_prefect.py --flow code
  python pipeline_prefect.py --flow api
  python pipeline_prefect.py --flow mlflow
  python pipeline_prefect.py --flow cd
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
REPO_URL      = "https://github.com/zied121/MLOPS.git"
DOCKER_IMAGE  = "zied_convergen_mlops"
DOCKER_HUB    = "zied121"


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


# ============================================================================
# TASKS – Git
# ============================================================================

@task(name="clone-or-pull-repo")
def task_clone_repo(repo_url: str = REPO_URL):
    """Clone the repo if absent, pull if already cloned."""
    logger = get_run_logger()
    project_dir = os.path.basename(repo_url).replace(".git", "")
    if os.path.exists(os.path.join(project_dir, ".git")):
        logger.info(f"📥 Pulling latest changes in '{project_dir}' ...")
        result = _run(["git", "-C", project_dir, "pull"])
    else:
        logger.info(f"📦 Cloning repo: {repo_url}")
        result = _run(["git", "clone", repo_url])
    if result.returncode == 0:
        logger.info("✅ Repo up to date.")
    else:
        logger.warning(f"⚠️  Git error:\n{result.stderr}")
    return result.returncode


# ============================================================================
# TASKS – Infrastructure
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
# TASKS – Code quality
# ============================================================================

@task(name="format-code")
def task_format_code():
    """Auto-format project files with Black."""
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
    """Run Flake8 linter on project files only."""
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
    """Run Bandit security scanner on project files only."""
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
    """Install pytest and run unit tests from test_pipeline.py."""
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
    """Run a sample prediction."""
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
    """Start FastAPI server with Uvicorn in background."""
    logger = get_run_logger()
    logger.info(f"🌐 Starting FastAPI on http://{host}:{port} ...")
    _run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "-q"])
    cmd = [sys.executable, "-m", "uvicorn", "app:app",
           "--host", host, "--port", str(port), "--reload"]
    log_file = open("uvicorn.log", "w")
    subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
    logger.info(f"✅ API running  → http://{host}:{port}")
    logger.info(f"📖 Swagger docs → http://127.0.0.1:{port}/docs")
    return f"http://{host}:{port}"


# ============================================================================
# TASKS – MLflow  (Atelier 5)
# ============================================================================

@task(name="run-mlflow-experiment")
def task_run_mlflow_experiment(n_estimators: int = 100, max_depth=None, test_size: float = 0.2):
    """Run a full MLflow-tracked experiment."""
    logger = get_run_logger()
    logger.info("🧪 Starting MLflow experiment ...")
    _run([sys.executable, "-m", "pip", "install", "mlflow", "-q"])
    from model_pipeline import run_experiment
    metrics = run_experiment(
        filepath=DATA_PATH, n_estimators=n_estimators,
        max_depth=max_depth, test_size=test_size, model_path=MODEL_PATH,
    )
    logger.info(f"✅ Accuracy: {metrics['accuracy']*100:.2f}% | F1: {metrics['f1_score']:.4f}")
    logger.info("   UI → http://127.0.0.1:5000")
    return metrics


@task(name="start-mlflow-ui")
def task_start_mlflow_ui(host: str = "0.0.0.0", port: int = 5000):
    """Start the MLflow UI in background using SQLite backend."""
    logger = get_run_logger()
    logger.info(f"🌐 Starting MLflow UI on http://{host}:{port} ...")
    cmd = [sys.executable, "-m", "mlflow", "ui",
           "--backend-store-uri", "sqlite:///mlflow.db",
           "--host", host, "--port", str(port)]
    log_file = open("mlflow_ui.log", "w")
    subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
    logger.info(f"✅ MLflow UI → http://127.0.0.1:{port}")
    return f"http://127.0.0.1:{port}"


# ============================================================================
# TASKS – Docker / CD  (Atelier 6)
# ============================================================================

@task(name="docker-build")
def task_docker_build(image_name: str = DOCKER_IMAGE):
    """Build the Docker image from the Dockerfile."""
    logger = get_run_logger()
    logger.info(f"🐳 Building Docker image: {image_name} ...")
    result = _run(["docker", "build", "-t", image_name, "."])
    if result.returncode == 0:
        logger.info(f"✅ Image built: {image_name}")
    else:
        logger.error(f"❌ Build failed:\n{result.stderr}")
    return result.returncode


@task(name="docker-run")
def task_docker_run(image_name: str = DOCKER_IMAGE, port: int = 8000):
    """Run the Docker container locally."""
    logger = get_run_logger()
    logger.info(f"▶️  Running container: {image_name} on port {port} ...")
    _run(["docker", "rm", "-f", "churn-api"])
    result = _run(["docker", "run", "-d", "--name", "churn-api",
                   "-p", f"{port}:{port}", image_name])
    if result.returncode == 0:
        logger.info(f"✅ Container running → http://localhost:{port}")
        logger.info(f"📖 Swagger docs    → http://localhost:{port}/docs")
    else:
        logger.error(f"❌ Run failed:\n{result.stderr}")
    return result.returncode


@task(name="docker-tag")
def task_docker_tag(image_name: str = DOCKER_IMAGE, hub_user: str = DOCKER_HUB):
    """Tag the image for Docker Hub."""
    logger = get_run_logger()
    tag = f"{hub_user}/{image_name}:latest"
    logger.info(f"🏷️  Tagging: {image_name} → {tag}")
    result = _run(["docker", "tag", image_name, tag])
    if result.returncode == 0:
        logger.info(f"✅ Tagged: {tag}")
    else:
        logger.error(f"❌ Tag failed:\n{result.stderr}")
    return tag


@task(name="docker-push")
def task_docker_push(image_name: str = DOCKER_IMAGE, hub_user: str = DOCKER_HUB):
    """Push the image to Docker Hub."""
    logger = get_run_logger()
    tag = f"{hub_user}/{image_name}:latest"
    logger.info(f"📤 Pushing to Docker Hub: {tag} ...")
    result = _run(["docker", "push", tag])
    if result.returncode == 0:
        logger.info(f"✅ Pushed → https://hub.docker.com/r/{hub_user}/{image_name}")
    else:
        logger.error(f"❌ Push failed:\n{result.stderr}")
    return result.returncode


# ============================================================================
# FLOWS
# ============================================================================

@flow(name="setup", log_prints=True)
def flow_setup():
    """clone/pull repo only"""
    task_clone_repo()


@flow(name="code", log_prints=True)
def flow_code():
    """format → lint → security → unit tests"""
    task_format_code()
    task_code_quality()
    task_security_check()
    task_run_unit_tests()


@flow(name="train", log_prints=True)
def flow_train():
    """prepare data → train → save"""
    x_train, x_test, y_train, y_test = task_prepare_data()
    model = task_train_model(x_train, y_train)
    task_save_model(model)


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    """load → prepare data → evaluate → predict"""
    model = task_load_model()
    x_train, x_test, y_train, y_test = task_prepare_data()
    task_evaluate_model(model, x_test, y_test)
    task_predict(model)


@flow(name="api", log_prints=True)
def flow_api():
    """load model → start FastAPI server"""
    task_load_model()
    task_start_api()


@flow(name="mlflow", log_prints=True)
def flow_mlflow():
    """run MLflow experiment → start UI"""
    task_run_mlflow_experiment()
    task_start_mlflow_ui()


@flow(name="cd", log_prints=True)
def flow_cd():
    """build image → run locally → tag → push to Docker Hub"""
    task_docker_build()
    task_docker_run()
    task_docker_tag()
    task_docker_push()


@flow(name="all", log_prints=True)
def flow_all():
    """clone → install → code quality → data → train → save → evaluate"""
    task_clone_repo()
    task_install_dependencies()
    task_format_code()
    task_code_quality()
    task_security_check()
    task_run_unit_tests()
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
    "mlflow":   flow_mlflow,
    "cd":       flow_cd,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefect ML Pipeline – Atelier 6")
    parser.add_argument(
        "--flow",
        type=str,
        choices=list(FLOWS.keys()),
        default="all",
        help=f"Flow to run: {' | '.join(FLOWS.keys())}  (default: all)",
    )
    args = parser.parse_args()
    print(f"\n🚀 Starting flow: '{args.flow}'\n")
    FLOWS[args.flow]()
