import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import torch

from .baseline import fit_xgboost, predict_xgboost
from .data import TemporalSplit, load_data
from .metrics import calibrate_threshold, evaluate, save_confusion_matrix
from .models import CONVOLUTIONS, TransactionGNN
from .rolling import rolling_predictions
from .training import fit, predict


MODELS = (*CONVOLUTIONS, "XGBoost")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def run_model(
    name: str, data: TemporalSplit, args: argparse.Namespace, device: torch.device
) -> dict:
    torch.manual_seed(args.seed)
    best_epoch = None
    if name == "XGBoost":
        model = fit_xgboost(data.train, args.seed, args.threads)
        predict_split = predict_xgboost
    else:
        model = TransactionGNN(name, data.train.num_features)
        optimizer, criterion, best_epoch = fit(model, data, args.loss, args.epochs, device)
        predict_split = predict

    val_labels, val_scores = predict_split(model, data.validation)
    threshold = calibrate_threshold(val_labels, val_scores) if args.calibrate else 0.5
    validation = evaluate(val_labels, val_scores, threshold)
    print(f"Validation macro F1 {validation['macro_f1']:.4f} | threshold {threshold:.4f}")

    if args.rolling:
        labels, scores = rolling_predictions(
            model, data.train, data.test, optimizer, criterion, args.rolling_epochs, threshold
        )
    else:
        labels, scores = predict_split(model, data.test)

    return {
        "model": name,
        "evaluation": "rolling" if args.rolling else "static",
        "seed": args.seed,
        "device": "cpu" if name == "XGBoost" else str(device),
        "threads": args.threads,
        "loss": None if name == "XGBoost" else args.loss,
        "epochs": None if name == "XGBoost" else args.epochs,
        "selected_epoch": best_epoch,
        "rolling_epochs": args.rolling_epochs if args.rolling else None,
        "threshold_source": "validation_macro_f1" if args.calibrate else "fixed",
        "versions": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in ("torch", "torch-geometric", "numpy", "pandas", "scikit-learn", "xgboost")
            },
        },
        "validation": validation,
        "test": evaluate(labels, scores, threshold),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gnn-fraud", description="Train and evaluate models on the Elliptic graph."
    )
    parser.add_argument("--model", choices=[*MODELS, "all"], default="GraphSAGE")
    parser.add_argument("--epochs", type=positive_int, default=100, help="initial GNN training epochs")
    parser.add_argument("--loss", choices=["focal", "weighted_ce"], default="focal")
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rolling", action="store_true", help="score and then update at each test timestep")
    parser.add_argument("--rolling-epochs", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)

    if args.rolling and args.model == "XGBoost":
        parser.error("rolling evaluation is available for GNNs")
    device_name = args.device
    if device_name == "auto":
        device_name = "cpu" if args.model == "XGBoost" or not torch.cuda.is_available() else "cuda"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    torch.set_num_threads(args.threads)
    names = (args.model,)
    if args.model == "all":
        names = tuple(CONVOLUTIONS) if args.rolling else MODELS

    print(f"Loading Elliptic from {args.data_dir}")
    data = load_data(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"\n{name}")
        result = run_model(name, data, args, device)
        mode, metrics = result["evaluation"], result["test"]
        stem = args.output_dir / f"{name.lower()}-{mode}"
        stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        save_confusion_matrix(metrics["confusion_matrix"], f"{name} ({mode})", stem.with_suffix(".png"))
        print(
            f"Test macro F1 {metrics['macro_f1']:.4f} | PR-AUC {metrics['pr_auc']:.4f}"
            f" | illicit recall {metrics['illicit_recall']:.2%}"
            f" | false-positive rate {metrics['false_positive_rate']:.2%}"
        )
        print(f"Saved {stem.with_suffix('.json')} and {stem.with_suffix('.png')}")
