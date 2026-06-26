"""Predict battle outcome from two Pokédex numbers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from src.features import DATA_DIR, STAT_COLS, _build_type_effect_matrix, _evolution_features, _load_stats, _mca_type_components

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "catboost.cbm"


def _pokemon_row(dex: int) -> pd.Series:
    stats = _load_stats()
    mca = _mca_type_components(stats)
    evo = _evolution_features(stats)
    base = stats.set_index("#")
    enriched = pd.concat([mca.set_index(stats["#"]), base, evo], axis=1)
    if dex not in enriched.index:
        raise SystemExit(f"Unknown Pokédex number: {dex}")
    return enriched.loc[dex]


def _matchup_features(first: int, second: int) -> pd.DataFrame:
    effect_matrix = _build_type_effect_matrix()
    a = _pokemon_row(first)
    b = _pokemon_row(second)

    feature_cols = (
        list(_mca_type_components(_load_stats()).columns)
        + STAT_COLS
        + ["is_Legendary", "is_Mega", "is_baby", "Stage"]
    )

    row: dict[str, float] = {}
    for col in feature_cols:
        row[f"{col}_x"] = float(a[col])
        row[f"{col}_y"] = float(b[col])
    for stat in STAT_COLS:
        row[f"{stat}_Delta"] = float(a[stat] - b[stat])

    mtype_a = a["Type_1"] + a["Type_2"]
    mtype_b = b["Type_1"] + b["Type_2"]

    def eff(att_mtype: str, d1: str, d2: str) -> float:
        e1 = effect_matrix.at[d1, att_mtype] if d1 in effect_matrix.index and att_mtype in effect_matrix.columns else 1.0
        e2 = effect_matrix.at[d2, att_mtype] if d2 in effect_matrix.index and att_mtype in effect_matrix.columns else 1.0
        return float(e1 * e2)

    row["First_Attacker_Eff"] = eff(mtype_a, b["Type_1"], b["Type_2"])
    row["Second_Attacker_Eff"] = eff(mtype_b, a["Type_1"], a["Type_2"])
    row["First_Speed"] = float(a["Speed"] >= b["Speed"])
    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict 1v1 battle outcome.")
    parser.add_argument("--a", type=int, required=True, help="First Pokédex number")
    parser.add_argument("--b", type=int, required=True, help="Second Pokédex number")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found. Run: python -m src.train")

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    features = _matchup_features(args.a, args.b)
    p_first = float(model.predict_proba(features)[0, 1])

    stats = _load_stats().set_index("#")
    name_a = stats.loc[args.a, "Name"]
    name_b = stats.loc[args.b, "Name"]
    print(f"{name_a.title()} (#{args.a}) vs {name_b.title()} (#{args.b})")
    print(f"P(first wins) = {p_first:.1%} · P(second wins) = {1 - p_first:.1%}")


if __name__ == "__main__":
    main()
