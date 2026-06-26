# Train three classifiers on the same 70-feature holdout split.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def _score(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    preds = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "log_loss": round(float(log_loss(y_true, proba)), 4),
    }


def train_logistic(x_train, y_train, x_test, y_test):
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=RANDOM_SEED)),
    ])
    pipe.fit(x_train, y_train)
    proba = pipe.predict_proba(x_test)[:, 1]
    return _score(y_test.to_numpy(), proba)


def train_catboost(x_train, y_train, x_test, y_test):
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
    out = _score(y_test.to_numpy(), proba)
    out["model_path"] = "models/catboost.cbm"
    return out


def train_tabpfn(x_train, y_train, x_test, y_test):
    from tabpfn import TabPFNClassifier

    if len(x_train) > TABPFN_MAX_TRAIN:
        x_sub, _, y_sub, _ = train_test_split(
            x_train, y_train,
            train_size=TABPFN_MAX_TRAIN,
            stratify=y_train,
            random_state=RANDOM_SEED,
        )
        fit_rows = TABPFN_MAX_TRAIN
    else:
        x_sub, y_sub = x_train, y_train
        fit_rows = len(x_train)

    clf = TabPFNClassifier(device="cpu")
    clf.fit(x_sub.astype(np.float32).to_numpy(), y_sub.to_numpy())
    proba = clf.predict_proba(x_test.astype(np.float32).to_numpy())[:, 1]
    out = _score(y_test.to_numpy(), proba)
    out["fit_rows"] = fit_rows
    return out


def _print_table(benchmarks: dict[str, dict | None]) -> None:
    print("\nHoldout (P first wins)")
    print(f"{'model':<22} {'acc':>8} {'auc':>8} {'logloss':>8}")
    print("-" * 48)
    for name in ("logistic_regression", "catboost", "tabpfn"):
        m = benchmarks.get(name)
        if m is None:
            print(f"{name:<22} {'n/a':>8} {'n/a':>8} {'n/a':>8}")
        elif "error" in m:
            print(f"{name:<22}  skipped")
        else:
            print(f"{name:<22} {m['accuracy']:>8.4f} {m['roc_auc']:>8.4f} {m['log_loss']:>8.4f}")


def main() -> None:
    _load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tabpfn", action="store_true")
    parser.add_argument("--test-size", type=float, default=HOLDOUT_FRAC)
    args = parser.parse_args()

    labeled = labeled_combat_frame()
    x_train, y_train, x_test, y_test = train_test_split_holdout(
        labeled, test_size=args.test_size, random_state=RANDOM_SEED
    )
    print(f"{len(x_train):,} train / {len(x_test):,} holdout / {x_train.shape[1]} features")

    benchmarks: dict[str, dict | None] = {}
    benchmarks["logistic_regression"] = train_logistic(x_train, y_train, x_test, y_test)
    benchmarks["catboost"] = train_catboost(x_train, y_train, x_test, y_test)

    benchmarks["tabpfn"] = None
    if not args.skip_tabpfn:
        try:
            benchmarks["tabpfn"] = train_tabpfn(x_train, y_train, x_test, y_test)
        except Exception as exc:  # noqa: BLE001
            benchmarks["tabpfn"] = {"error": str(exc)}

    _print_table(benchmarks)

    results = {
        "split": {
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "test_fraction": args.test_size,
            "random_seed": RANDOM_SEED,
            "features": x_train.shape[1],
        },
        "benchmarks": benchmarks,
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {OUTPUTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
