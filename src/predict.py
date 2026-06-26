"""Predict battle outcome from two Pokémon ids (pokemon.csv `#` column)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from catboost import CatBoostClassifier

from src.features import build_matchup_features, pokemon_name

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "catboost.cbm"


def predict_matchup(first_id: int, second_id: int) -> float:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Run python -m src.train first.")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    features = build_matchup_features(first_id, second_id)
    return float(model.predict_proba(features)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P(first Pokémon wins). Ids are the `#` column in data/raw/pokemon.csv."
    )
    parser.add_argument("--a", type=int, required=True, help="First Pokémon id")
    parser.add_argument("--b", type=int, required=True, help="Second Pokémon id")
    args = parser.parse_args()

    try:
        p_first = predict_matchup(args.a, args.b)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)

    print(f"{pokemon_name(args.a)} (#{args.a}) vs {pokemon_name(args.b)} (#{args.b})")
    print(f"P(first wins) = {p_first:.1%}   P(second wins) = {1 - p_first:.1%}")


if __name__ == "__main__":
    main()
