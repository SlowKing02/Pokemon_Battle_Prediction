# Pokémon Battle Prediction

Binary classifier on 50k labeled 1v1 battles. Features come from a 2018 grad-school script (MCA on types, stat deltas, type chart); CatBoost is the main model. TabPFN runs on the same holdout if you add a token.

CSVs merged from the old `Pokemon_Dataset` repo (2026 rebuild).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Features

70 columns per combat row. See [docs/FEATURES.md](docs/FEATURES.md) for the full list.

```bash
python -c "from src.features import feature_column_names; print(len(feature_column_names()))"
```

## Train

```bash
python -m src.train                 # CatBoost + TabPFN when TABPFN_TOKEN is set
python -m src.train --skip-tabpfn   # CatBoost only (~30s on CPU)
python -m unittest tests.test_pipeline -v
```

Holdout: stratified 85/15 from the 50k labeled rows (`Test_Set=0`). The 2,080 `Test_Set=1` rows have no winner label (Kaggle export); they are not used for training or metrics.

| Output | Contents |
|--------|----------|
| `models/catboost.cbm` | Saved classifier |
| `outputs/metrics.json` | Holdout metrics |

Checked-in example (CatBoost, 70 features): `outputs/metrics.example.json` (98.52% accuracy, 0.999 ROC-AUC).

### TabPFN (optional)

TabPFN fits on up to 10k stratified train rows (library cap), then scores the same 7,500-row holdout as CatBoost.

1. Accept license at https://ux.priorlabs.ai
2. Put the API key in `.env` as `TABPFN_TOKEN` (see `.env.example`)
3. Run `python -m src.train`

Without a token, training still runs CatBoost and writes `"tabpfn": null` in `metrics.json`.

## Predict

Ids are the `#` column in `data/raw/pokemon.csv`, not national dex numbers.

```bash
python -m src.predict --a 163 --b 7
# Mewtwo (#163) vs Charizard (#7)
```

## Data (`data/raw/`)

| File | Role |
|------|------|
| `combats_test.csv` | Combats + `Test_Set` flag |
| `pokemon.csv` | Stats, types |
| `chart.csv` | Type effectiveness |
| `pokemon_species.csv` | Evolution chains |

## Layout

```text
src/features.py    combat matrix, holdout split, single-matchup rows
docs/FEATURES.md   column dictionary
src/train.py       CatBoost train + optional TabPFN compare
src/predict.py     CLI
tests/             split and feature smoke tests
legacy/            2018 scripts (reference only)
```

## Legacy

`legacy/Pokemon_Battle_Match.py` is the original RF/Keras pipeline. `legacy/CatBoost.py` is an early CatBoost run on exported CSVs. Use `src/` for new work.

## License

MIT. Fan/research use; not affiliated with Nintendo/Creatures/Game Freak.
