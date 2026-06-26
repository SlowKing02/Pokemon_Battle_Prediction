# Pokémon Battle Prediction

Binary classifier on 50k labeled 1v1 battles. Feature engineering from the 2018 script (types, stat deltas, type chart, MCA on types); CatBoost classifier; optional TabPFN comparison on the same holdout.

Grad-school project, rebuilt 2026. CSVs merged from the old `Pokemon_Dataset` repo.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Feature engineering

70 columns per combat: MCA-compressed types, raw stats and meta for both sides, stat deltas, type-chart effectiveness, and matchup derivations (BST/offense/bulk deltas, speed ratio, shared-type flag, etc.).

Full column list and rationale: [docs/FEATURES.md](docs/FEATURES.md).

```bash
python -c "from src.features import feature_column_names; print(len(feature_column_names()))"
```

## Train and evaluate

```bash
python -m src.train              # CatBoost + TabPFN (if TABPFN_TOKEN set)
python -m src.train --skip-tabpfn   # CatBoost only, ~30s CPU
python -m unittest tests.test_pipeline -v
```

**Split:** stratified 85/15 from the 50k labeled rows (`Test_Set=0`). The 2,080 `Test_Set=1` rows have no winner label (Kaggle submission export); they are ignored for metrics.

**Outputs:**

| Path | Contents |
|------|----------|
| `models/catboost.cbm` | Saved classifier |
| `outputs/metrics.json` | Holdout metrics for CatBoost and TabPFN |

Example metrics (CatBoost only): see `outputs/metrics.example.json` (~98.5% accuracy, 0.999 ROC-AUC on holdout, 70 features).

### TabPFN comparator

TabPFN fits on at most 10,000 stratified train rows (library limit), evaluates on the **same** 7,500-row holdout as CatBoost. Requires a free Prior Labs token:

1. Accept license at https://ux.priorlabs.ai  
2. Copy API key → `TABPFN_TOKEN` in `.env` (see `.env.example`)  
3. Re-run `python -m src.train`

Training prints a side-by-side table:

```text
model        accuracy    roc_auc   log_loss
--------------------------------------------
catboost         0.9821     0.9987     0.0491
tabpfn           …          …          …
```

Without a token, CatBoost still runs; `metrics.json` records the TabPFN error string.

## Predict

Ids are the `#` column in `data/raw/pokemon.csv` (not national dex).

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
src/features.py   build combat matrix, train/holdout helpers, single-matchup rows
docs/FEATURES.md   column dictionary
src/train.py      CatBoost train + TabPFN compare
src/predict.py    CLI
tests/            split + feature smoke tests
legacy/           2018 monolith scripts (reference)
```

## Legacy

`legacy/Pokemon_Battle_Match.py` — original RF/keras pipeline. `legacy/CatBoost.py` — early CatBoost on exported CSVs. Do not use for new runs.

## License

MIT. Fan/research use; not affiliated with Nintendo/Creatures/Game Freak.
