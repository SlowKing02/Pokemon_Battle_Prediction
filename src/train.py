"""Train CatBoost, evaluate on holdout, compare to TabPFN."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import labeled_combat_frame, train_test_split_holdout

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
TABPFN_MAX_TRAIN = 10_000
HOLDOUT_FRAC = 0.15
RANDOM_SEED = 42


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    preds = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "log_loss": round(float(log_loss(y_true, proba)), 4),
    }


def train_catboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_SEED,
        verbose=False,
    )
    model.fit(x_train, y_train, eval_set=(x_test, y_test), use_best_model=True)
    proba = model.predict_proba(x_test)[:, 1]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODELS_DIR / "catboost.cbm"))
    out = _metrics(y_test.to_numpy(), proba)
    out["model_path"] = "models/catboost.cbm"
    return out


def train_tabpfn(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    from tabpfn import TabPFNClassifier

    train_rows = len(x_train)
    if train_rows > TABPFN_MAX_TRAIN:
        x_sub, _, y_sub, _ = train_test_split(
            x_train,
            y_train,
            train_size=TABPFN_MAX_TRAIN,
            stratify=y_train,
            random_state=RANDOM_SEED,
        )
        fit_rows = TABPFN_MAX_TRAIN
    else:
        x_sub, y_sub = x_train, y_train
        fit_rows = train_rows

    clf = TabPFNClassifier(device="cpu")
    clf.fit(x_sub.astype(np.float32).to_numpy(), y_sub.to_numpy())
    proba = clf.predict_proba(x_test.astype(np.float32).to_numpy())[:, 1]
    out = _metrics(y_test.to_numpy(), proba)
    out["fit_rows"] = fit_rows
    out["note"] = f"TabPFN fit on {fit_rows:,} stratified train rows (library cap {TABPFN_MAX_TRAIN:,})"
    return out


def _print_comparison(catboost: dict, tabpfn: dict | None) -> None:
    print("\nHoldout comparison (first Pokémon wins)")
    print(f"{'model':<12} {'accuracy':>10} {'roc_auc':>10} {'log_loss':>10}")
    print("-" * 44)
    print(
        f"{'catboost':<12} {catboost['accuracy']:>10.4f} "
        f"{catboost['roc_auc']:>10.4f} {catboost['log_loss']:>10.4f}"
    )
    if tabpfn and "error" not in tabpfn:
        print(
            f"{'tabpfn':<12} {tabpfn['accuracy']:>10.4f} "
            f"{tabpfn['roc_auc']:>10.4f} {tabpfn['log_loss']:>10.4f}"
        )
    elif tabpfn and "error" in tabpfn:
        print(f"{'tabpfn':<12} skipped ({tabpfn['error'][:60]}…)")


def main() -> None:
    _load_env()
    parser = argparse.ArgumentParser(description="Train CatBoost and compare TabPFN on holdout.")
    parser.add_argument("--skip-tabpfn", action="store_true", help="CatBoost only.")
    parser.add_argument("--test-size", type=float, default=HOLDOUT_FRAC, help="Holdout fraction.")
    args = parser.parse_args()

    print("Building features…")
    labeled = labeled_combat_frame()
    x_train, y_train, x_test, y_test = train_test_split_holdout(
        labeled, test_size=args.test_size, random_state=RANDOM_SEED
    )
    print(f"Train {len(x_train):,} · holdout {len(x_test):,} · {x_train.shape[1]} features")
    print("Test_Set=1 rows in the CSV are unlabeled Kaggle submits; not used here.")

    catboost_metrics = train_catboost(x_train, y_train, x_test, y_test)
    print("CatBoost done.", catboost_metrics)

    tabpfn_metrics: dict | None = None
    if not args.skip_tabpfn:
        print("TabPFN…")
        try:
            tabpfn_metrics = train_tabpfn(x_train, y_train, x_test, y_test)
            print("TabPFN done.", tabpfn_metrics)
        except Exception as exc:  # noqa: BLE001
            tabpfn_metrics = {"error": str(exc)}
            print("TabPFN skipped:", exc)

    _print_comparison(catboost_metrics, tabpfn_metrics)

    results: dict[str, object] = {
        "split": {
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "test_fraction": args.test_size,
            "random_seed": RANDOM_SEED,
            "features": x_train.shape[1],
        },
        "catboost": catboost_metrics,
        "tabpfn": tabpfn_metrics,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "metrics.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
