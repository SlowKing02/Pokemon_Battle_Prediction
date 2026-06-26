"""Train CatBoost baseline and TabPFN benchmark on official holdout."""

from __future__ import annotations

import argparse
import json
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


def _metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    preds = (proba >= 0.5).astype(int)
    out = {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
    }
    if len(np.unique(y_true)) > 1:
        out["log_loss"] = round(float(log_loss(y_true, proba)), 4)
    return out


def train_catboost(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
    )
    model.fit(x_train, y_train, eval_set=(x_test, y_test), use_best_model=True)
    proba = model.predict_proba(x_test)[:, 1]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODELS_DIR / "catboost.cbm"))
    return _metrics(y_test.to_numpy(), proba)


def train_tabpfn(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    from tabpfn import TabPFNClassifier

    if len(x_train) > TABPFN_MAX_TRAIN:
        x_sub, _, y_sub, _ = train_test_split(
            x_train,
            y_train,
            train_size=TABPFN_MAX_TRAIN,
            stratify=y_train,
            random_state=42,
        )
    else:
        x_sub, y_sub = x_train, y_train

    x_sub = x_sub.astype(np.float32)
    x_test_f = x_test.astype(np.float32)

    clf = TabPFNClassifier(device="cpu")
    clf.fit(x_sub.to_numpy(), y_sub.to_numpy())
    proba = clf.predict_proba(x_test_f.to_numpy())[:, 1]
    return _metrics(y_test.to_numpy(), proba)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train battle outcome models.")
    parser.add_argument("--skip-tabpfn", action="store_true", help="Skip TabPFN benchmark (faster).")
    args = parser.parse_args()

    print("Building features…")
    labeled = labeled_combat_frame()
    x_train, y_train, x_test, y_test = train_test_split_holdout(labeled)
    print(f"Train: {len(x_train):,} · Holdout: {len(x_test):,} · Features: {x_train.shape[1]}")
    print("(Test_Set=1 in source data is unlabeled Kaggle submission; not used for metrics.)")

    results: dict[str, object] = {
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "features": x_train.shape[1],
        "holdout": "stratified 15% from 50k labeled combats",
        "models": {},
    }

    print("Training CatBoost…")
    results["models"]["catboost"] = train_catboost(x_train, y_train, x_test, y_test)
    print("CatBoost:", results["models"]["catboost"])

    if not args.skip_tabpfn:
        print(f"Training TabPFN (train subsample ≤ {TABPFN_MAX_TRAIN:,})…")
        try:
            results["models"]["tabpfn"] = train_tabpfn(x_train, y_train, x_test, y_test)
            print("TabPFN:", results["models"]["tabpfn"])
        except Exception as exc:  # noqa: BLE001
            results["models"]["tabpfn"] = {"error": str(exc)}
            print("TabPFN failed:", exc)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "metrics.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
