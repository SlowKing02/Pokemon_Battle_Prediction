# Pokémon Battle Prediction

Supervised learning on **50,000 labeled 1v1 battles**: given two Pokédex IDs and their stats/types, predict which combatant wins. Includes a **CatBoost** baseline and **TabPFN** benchmark on a stratified 15% holdout from the labeled set.

**GitHub:** [SlowKing02/Pokemon_Battle_Prediction](https://github.com/SlowKing02/Pokemon_Battle_Prediction)

## Problem

Pairwise binary classification: `First_Winner = 1` if the first listed Pokémon wins. Features combine stat deltas, type effectiveness (attack chart), MCA-compressed types, evolution stage, and legendary flags — the same feature philosophy as the 2018 prototype, rebuilt as a reproducible pipeline.

## Data

Merged from the former [Pokemon_Dataset](https://github.com/SlowKing02/Pokemon_Dataset) repo into `data/raw/`:

| File | Role |
|------|------|
| `combats_test.csv` | Labeled combats (`Test_Set=0`) + unlabeled Kaggle rows (`Test_Set=1`) |
| `pokemon.csv` | Base stats and types |
| `chart.csv` | Type effectiveness matrix |
| `pokemon_species.csv` | Evolution chain metadata |

Source: Kaggle-style Pokémon combat prediction task (circa 2018).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train CatBoost + TabPFN benchmark (~2–5 min on CPU)
python -m src.train

# Predict a matchup (after training)
python -m src.predict --a 25 --b 6
```

Metrics land in `outputs/metrics.json`; CatBoost model in `models/catboost.cbm`.

**TabPFN:** set `TABPFN_TOKEN` (see `.env.example`) after accepting the license at [ux.priorlabs.ai](https://ux.priorlabs.ai). Without a token, CatBoost still trains; TabPFN is skipped with a note in `metrics.json`.

## Layout

```text
data/raw/           # merged CSVs
src/
  features.py       # vectorized feature engineering
  train.py          # CatBoost + TabPFN on official holdout
  predict.py        # CLI win probability
models/             # trained artifacts (gitignored)
outputs/            # metrics.json
legacy/             # 2018 scripts (reference)
```

## TabPFN note

TabPFN is capped at **10,000 training rows** by the library; training uses a stratified subsample from the train split, evaluated on the same 15% holdout as CatBoost. The source file’s `Test_Set=1` rows have no winner labels (Kaggle submission format).

## Legacy

Original monolith scripts (`Pokemon_Battle_Match.py`, `CatBoost.py`) remain at repo root for reference; use `src/` for all new work.

## License

MIT · Pokémon data is fan/research use; not affiliated with Nintendo/Creatures/Game Freak.
