# Pokémon Battle Prediction

Grad-school side project from 2018, cleaned up in 2026. Given two Pokémon and their base stats/types, predict who wins a simulated 1v1.

The Kaggle-style dataset has 50,000 labeled fights and ~800 species (gens 1–6, megas included). Each row is just `(first_id, second_id, winner)`. No moves, no items, no abilities in the combat file. So most of the work is encoding what you *can* know from stats and the type chart before you touch a model.

## Results (holdout, seed 42)

Stratified 85/15 split on the 50k labeled rows. Same 70 features for every model.

| Model | Accuracy | ROC-AUC | Log loss |
|-------|----------|---------|----------|
| Logistic regression | 91.85% | 0.961 | 0.234 |
| CatBoost | **98.52%** | **0.999** | **0.044** |
| TabPFN | 97.60% | 0.998 | 0.062 |

TabPFN needs a Prior Labs token and a one-time license at [ux.priorlabs.ai](https://ux.priorlabs.ai). On CPU, set `TABPFN_ALLOW_CPU_LARGE_DATASET=1` in `.env` (see `.env.example`); `train.py` sets this automatically.

CatBoost is what `predict.py` loads. The logistic gap (~7pp) is the interesting part: the features already explain a lot; boosting picks up type interactions the linear model misses.

Full write-up: [docs/BENCHMARK.md](docs/BENCHMARK.md). Feature list: [docs/FEATURES.md](docs/FEATURES.md). Game/dataset notes: [docs/CONTEXT.md](docs/CONTEXT.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --skip-tabpfn
python -m src.predict --a 163 --b 7   # mewtwo vs charizard
python -m unittest tests.test_pipeline -v
```

## What's in the repo

```text
src/features.py    70-column combat matrix
src/train.py       logistic + CatBoost + optional TabPFN on same holdout
src/predict.py     load CatBoost, score one matchup
docs/              features, benchmark notes, pokemon context
legacy/            original 2018 scripts
```

## Data files (`data/raw/`)

| File | Notes |
|------|-------|
| `combats_test.csv` | 50k labeled + 2,080 unlabeled Kaggle rows (`Test_Set=1`, no winner) |
| `pokemon.csv` | Base stats, types; `#` column is the id used everywhere |
| `chart.csv` | Type effectiveness multipliers |
| `pokemon_species.csv` | Evolution chains for stage features |
| `pokedex.csv` | Extra fields (abilities, type resistances); not wired in yet |

## License

MIT. Fan project; not affiliated with Nintendo/Creatures/Game Freak.
