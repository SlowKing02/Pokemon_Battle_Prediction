# pokemon_battle_prediction

MSA side project (2018), rebuilt 2026. Given two Pokémon species and their base stats, predict who wins a simulated 1v1.

The Kaggle export has 50,000 labeled fights and ~800 species through gen 6 (megas included). Each row is `(first_id, second_id, winner)` only. No moves, items, or abilities in the combat file, so most of the signal has to come from typings, stat gaps, and the type chart.

## Holdout results

15% stratified holdout · seed 42 · 70 features · 42,500 train / 7,500 test

| Model | Accuracy | ROC-AUC | Log loss |
|-------|----------|---------|----------|
| Logistic regression | 91.85% | 0.961 | 0.234 |
| CatBoost | **98.52%** | **0.999** | **0.044** |
| TabPFN | 97.60% | 0.998 | 0.062 |

Logistic regression at ~92% means the engineered columns already carry most of the story. CatBoost picks up another ~7pp from type/stat interactions. TabPFN trained on 10k rows (library cap) and scored the same holdout; it trails CatBoost slightly on this feature set.

`predict.py` loads the CatBoost model. TabPFN is optional; see `.env.example` for token and CPU settings.

Details: [docs/BENCHMARK.md](docs/BENCHMARK.md) · columns: [docs/FEATURES.md](docs/FEATURES.md) · dataset notes: [docs/CONTEXT.md](docs/CONTEXT.md)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train              # all three models if TABPFN_TOKEN set
python -m src.train --skip-tabpfn   # logistic + CatBoost only (~1 min)
python -m src.predict --a 163 --b 7
python -m unittest tests.test_pipeline -v
```

Ids in `predict.py` are the `#` column in `data/raw/pokemon.csv`, not national dex numbers.

## Repo layout

```text
src/features.py   build the 70-column matrix
src/train.py      benchmark harness (logistic, CatBoost, TabPFN)
src/predict.py    score one matchup
docs/             feature list, benchmark notes, Pokémon context
legacy/           2018 RF/Keras scripts for reference
data/raw/         CSVs (combats, stats, type chart, species)
```

## Data

| File | Role |
|------|------|
| `combats_test.csv` | Labeled fights + 2,080 unlabeled Kaggle rows (`Test_Set=1`) |
| `pokemon.csv` | Base stats and types |
| `chart.csv` | Type effectiveness multipliers |
| `pokemon_species.csv` | Evolution chains (stage feature) |
| `pokedex.csv` | Abilities and resistances; not used in the current pipeline |

## License

MIT. Fan/research use. Not affiliated with Nintendo, Creatures, or Game Freak.
