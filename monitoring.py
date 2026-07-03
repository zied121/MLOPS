"""
monitoring.py
-------------
Atelier 7 – Monitoring MLOps avec MLflow + Elasticsearch + Kibana

Récupère les métriques MLflow et les envoie vers Elasticsearch
pour visualisation dans Kibana.

Usage
-----
  python monitoring.py
"""

import datetime
import time

import mlflow
from elasticsearch import Elasticsearch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME     = "churn-prediction"
ES_HOST             = "http://localhost:9200"
ES_INDEX            = "mlflow-metrics"

# ---------------------------------------------------------------------------
# Connexion Elasticsearch
# ---------------------------------------------------------------------------

def get_es_client():
    """Créer et retourner un client Elasticsearch."""
    es = Elasticsearch(ES_HOST)
    if es.ping():
        print(f"✅ Connecté à Elasticsearch : {ES_HOST}")
    else:
        raise ConnectionError(f"❌ Impossible de se connecter à Elasticsearch : {ES_HOST}")
    return es


# ---------------------------------------------------------------------------
# Créer l'index Elasticsearch
# ---------------------------------------------------------------------------

def create_index(es: Elasticsearch):
    """Créer l'index mlflow-metrics s'il n'existe pas."""
    if es.indices.exists(index=ES_INDEX):
        print(f"📋 Index '{ES_INDEX}' existe déjà.")
        return

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "@timestamp":          {"type": "date"},
                "run_id":              {"type": "keyword"},
                "run_name":            {"type": "keyword"},
                "experiment_id":       {"type": "keyword"},
                "experiment_name":     {"type": "keyword"},
                "status":              {"type": "keyword"},
                "duration_seconds":    {"type": "float"},
                "metric_accuracy":     {"type": "float"},
                "metric_precision":    {"type": "float"},
                "metric_recall":       {"type": "float"},
                "metric_f1_score":     {"type": "float"},
                "metric_test_samples": {"type": "float"},
                "param_n_estimators":  {"type": "keyword"},
                "param_max_depth":     {"type": "keyword"},
                "param_model_type":    {"type": "keyword"},
                "param_train_samples": {"type": "keyword"},
                "params":              {"type": "object"},
                "metrics":             {"type": "object"},
            }
        }
    }

    es.indices.create(index=ES_INDEX, body=mapping)
    print(f"✅ Index '{ES_INDEX}' créé.")


# ---------------------------------------------------------------------------
# Récupérer les runs MLflow
# ---------------------------------------------------------------------------

def get_mlflow_runs():
    """Récupérer tous les runs de l'expérience MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"⚠️  Expérience '{EXPERIMENT_NAME}' introuvable.")
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    print(f"📊 {len(runs)} runs trouvés dans '{EXPERIMENT_NAME}'")
    return runs, experiment


# ---------------------------------------------------------------------------
# Envoyer les métriques vers Elasticsearch
# ---------------------------------------------------------------------------

def send_to_elasticsearch(es: Elasticsearch, runs, experiment):
    """Envoyer chaque run MLflow comme document Elasticsearch."""
    sent = 0
    for run in runs:
        info    = run.info
        metrics = run.data.metrics
        params  = run.data.params

        # Durée du run
        duration = 0
        if info.end_time and info.start_time:
            duration = (info.end_time - info.start_time) / 1000.0

        # Document Elasticsearch
        doc = {
            "@timestamp":          datetime.datetime.utcfromtimestamp(
                                       info.start_time / 1000.0
                                   ).isoformat() + "Z",
            "run_id":              info.run_id,
            "run_name":            info.run_name or "unnamed",
            "experiment_id":       info.experiment_id,
            "experiment_name":     experiment.name,
            "status":              info.status,
            "duration_seconds":    round(duration, 2),

            # Métriques individuelles (pour filtres Kibana)
            "metric_accuracy":     metrics.get("accuracy",     0),
            "metric_precision":    metrics.get("precision",    0),
            "metric_recall":       metrics.get("recall",       0),
            "metric_f1_score":     metrics.get("f1_score",     0),
            "metric_test_samples": metrics.get("test_samples", 0),

            # Hyperparamètres
            "param_n_estimators":  params.get("n_estimators", ""),
            "param_max_depth":     params.get("max_depth",    ""),
            "param_model_type":    params.get("model_type",   ""),
            "param_train_samples": params.get("train_samples",""),

            # Objets complets (pour recherche avancée)
            "params":  params,
            "metrics": metrics,
        }

        es.index(index=ES_INDEX, id=info.run_id, body=doc)
        print(f"  ✅ Run {info.run_id[:8]} → ES | acc={metrics.get('accuracy', 'N/A')}")
        sent += 1

    print(f"\n📤 {sent} runs envoyés vers Elasticsearch index '{ES_INDEX}'")
    return sent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Atelier 7 – Monitoring MLflow → Elasticsearch")
    print("=" * 60)

    # 1. Connexion ES
    es = get_es_client()

    # 2. Créer index
    create_index(es)

    # 3. Récupérer runs MLflow
    result = get_mlflow_runs()
    if not result:
        print("Aucun run à envoyer.")
        return
    runs, experiment = result

    # 4. Envoyer vers ES
    send_to_elasticsearch(es, runs, experiment)

    print("\n✅ Monitoring terminé !")
    print(f"   MLflow UI → http://localhost:5000")
    print(f"   Kibana    → http://localhost:5601")
    print(f"   ES index  → {ES_HOST}/{ES_INDEX}/_search")


if __name__ == "__main__":
    main()
