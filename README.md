# Pokémon Battle Prediction

Binary classifier on 50k labeled 1v1 battles: stats, types, and a type chart in; winner out. CatBoost baseline. Optional TabPFN comparison if you have a Prior Labs token.

Grad-school side project (2018); rebuilt the pipeline in 2026 and merged the CSVs that used to live in a separate dataset repo.

## Data

All under `data/raw/`:

| File | Notes |
|------|--------|
| `combats_test.csv` | 50k labeled rows (`Test_Set=0`); 2k Kaggle submit rows with no winner (`Test_Set=1`) |
| `pokemon.csv` | Stats and types (`#` is row id in this file, not national dex) |
| `chart.csv` | Type matchup multipliers |
| `pokemon_species.csv` | Evolution metadata |

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
python -m src.predict --a 163 --b 7   # after train; ids from pokemon.csv
```

Holdout is a random 15% split from the 50k labeled combats (not `Test_Set=1`, which has no labels). Metrics: `outputs/metrics.json`. Model: `models/catboost.cbm`.

TabPFN (optional): cap 10k train rows by library limits. Set `TABPFN_TOKEN` in `.env` — see `.env.example` and https://ux.priorlabs.ai

## Layout

```text
src/features.py   # ETL (MCA types, stat deltas, type chart) — vectorized
src/train.py      # CatBoost + optional TabPFN
src/predict.py    # CLI
legacy/           # 2018 monolith scripts
```

## Legacy

`legacy/Pokemon_Battle_Match.py` and `legacy/CatBoost.py` are the original scripts. Use `src/` for anything new.

## License

MIT. Fan/research use; not affiliated with Nintendo/Creatures/Game Freak.
