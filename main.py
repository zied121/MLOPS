"""
main.py
-------
Entry point for the Customer Churn ML pipeline.
Executes the modular functions from model_pipeline.py.

Usage (CLI)
-----------
Run the full pipeline:
    python main.py --all

Run individual steps:
    python main.py --prepare
    python main.py --train
    python main.py --evaluate
    python main.py --save
    python main.py --load

Custom data path:
    python main.py --all --data Churn_Modelling.csv
"""

import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Customer Churn ML Pipeline – Atelier 2"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="Churn_Modelling.csv",
        help="Path to the CSV dataset (default: Churn_Modelling.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="classifier.joblib",
        help="Path to save/load the model (default: classifier.joblib)",
    )
    parser.add_argument("--all", action="store_true", help="Run the full pipeline")
    parser.add_argument("--prepare", action="store_true", help="Prepare data only")
    parser.add_argument("--train", action="store_true", help="Train model only")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model only")
    parser.add_argument("--save", action="store_true", help="Save model only")
    parser.add_argument("--load", action="store_true", help="Load model only")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # Default: run full pipeline if no flag given
    run_all = args.all or not any(
        [args.prepare, args.train, args.evaluate, args.save, args.load]
    )

    x_train = x_test = y_train = y_test = model = None

    # ── Step 1 : Prepare data ─────────────────────────────────────────────
    if run_all or args.prepare:
        print("\n" + "=" * 50)
        print("  STEP 1 – Prepare Data")
        print("=" * 50)
        x_train, x_test, y_train, y_test = prepare_data(args.data)

    # ── Step 2 : Train model ──────────────────────────────────────────────
    if run_all or args.train:
        print("\n" + "=" * 50)
        print("  STEP 2 – Train Model")
        print("=" * 50)
        if x_train is None:
            # Need data first
            x_train, x_test, y_train, y_test = prepare_data(args.data)
        model = train_model(x_train, y_train)

    # ── Step 3 : Evaluate model ───────────────────────────────────────────
    if run_all or args.evaluate:
        print("\n" + "=" * 50)
        print("  STEP 3 – Evaluate Model")
        print("=" * 50)
        if model is None:
            print("[main] No model in memory – loading from disk...")
            model = load_model(args.model)
        if x_test is None:
            _, x_test, _, y_test = prepare_data(args.data)
        metrics = evaluate_model(model, x_test, y_test, show_plot=True)
        print(f"[main] Final Accuracy: {metrics['accuracy']*100:.2f}%")

    # ── Step 4 : Save model ───────────────────────────────────────────────
    if run_all or args.save:
        print("\n" + "=" * 50)
        print("  STEP 4 – Save Model")
        print("=" * 50)
        if model is None:
            print("[main] No model to save – train first.")
        else:
            save_model(model, args.model)

    # ── Step 5 : Load model (demo) ────────────────────────────────────────
    if run_all or args.load:
        print("\n" + "=" * 50)
        print("  STEP 5 – Load Model")
        print("=" * 50)
        loaded_model = load_model(args.model)
        print(f"[main] Loaded model type: {type(loaded_model).__name__}")

    print("\n✅  Pipeline finished.\n")


if __name__ == "__main__":
    main()
