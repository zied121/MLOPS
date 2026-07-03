"""
deploiement_prefect.py
----------------------
Registers Prefect deployments with daily schedules.
Requires a running Prefect server (prefect server start).

Setup (3 terminals)
-------------------
Terminal 1 – Start the Prefect server:
    prefect server start
    → UI available at http://localhost:4200

Terminal 2 – Register deployments and start a worker:
    prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
    python deploiement_prefect.py

Terminal 3 – Trigger a deployment manually:
    prefect deployment ls
    prefect deployment run 'all/ml-pipeline-all'
    prefect deployment run 'train/ml-pipeline-train'
    prefect deployment run 'evaluate/ml-pipeline-evaluate'
    prefect deployment run 'code/ml-pipeline-code'
"""

from prefect import serve


from pipeline_prefect import flow_all, flow_train, flow_evaluate, flow_code

if __name__ == "__main__":
    print("📋 Registering Prefect deployments ...")

    # ── Deployment 1 : full pipeline – every day at 02:00 ─────────────────
    deploy_all = flow_all.to_deployment(
        name="ml-pipeline-all",
        tags=["mlops", "full-pipeline"],
        cron="0 2 * * *",        # 02:00 every day
    )

    # ── Deployment 2 : training only – every day at 03:00 ─────────────────
    deploy_train = flow_train.to_deployment(
        name="ml-pipeline-train",
        tags=["mlops", "training"],
        cron="0 3 * * *",        # 03:00 every day
    )

    # ── Deployment 3 : evaluation only – every day at 04:00 ───────────────
    deploy_evaluate = flow_evaluate.to_deployment(
        name="ml-pipeline-evaluate",
        tags=["mlops", "evaluation"],
        cron="0 4 * * *",        # 04:00 every day
    )

    # ── Deployment 4 : code quality – every day at 01:00 ──────────────────
    deploy_code = flow_code.to_deployment(
        name="ml-pipeline-code",
        tags=["mlops", "code-quality"],
        cron="0 1 * * *",        # 01:00 every day
    )

    print("✅ Deployments registered. Starting worker — waiting for runs ...")
    print("   Open http://localhost:4200 to view the Prefect UI.\n")

    # serve() starts a local worker that picks up scheduled / manual runs
    serve(deploy_all, deploy_train, deploy_evaluate, deploy_code)
