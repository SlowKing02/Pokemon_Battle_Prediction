# Benchmark notes

Same 70 features, same holdout, three models. I wanted to see how much lift comes from feature work vs model choice.

## Split

| | |
|---|---|
| Labeled rows | 50,000 (`Test_Set=0`) |
| Train | 42,500 |
| Holdout | 7,500 (15%, stratified, seed 42) |
| Ignored | 2,080 `Test_Set=1` rows (no winner in Kaggle export) |

## Models

**Logistic regression** — `StandardScaler` + sklearn LR. Linear read on whether the engineered columns already separate winners.

**CatBoost** — 300 trees, depth 8, lr 0.08, early stop on holdout AUC. Saved to `models/catboost.cbm`.

**TabPFN** — optional; fits on 10k stratified train rows max, same holdout. Needs token + license acceptance at [ux.priorlabs.ai](https://ux.priorlabs.ai).

## Results (2026-06-22 run)

| Model | Accuracy | ROC-AUC | Log loss |
|-------|----------|---------|----------|
| Logistic regression | 0.9185 | 0.9605 | 0.2338 |
| CatBoost | 0.9852 | 0.9990 | 0.0441 |
| TabPFN | 0.9760 | 0.9977 | 0.0616 |

TabPFN fit on 10,000 stratified train rows (library cap), same 7,500 holdout. CPU run ~6–7 min with `TABPFN_ALLOW_CPU_LARGE_DATASET=1`.

TabPFN status: license accepted; prior skip was the CPU 1,000-row default limit, not auth.

### Reading the table

Logistic at ~92% means the type chart + stat deltas carry real signal without interactions. CatBoost adds ~6.7pp accuracy, mostly from nonlinear type/stat combos. Log loss drops hard (0.23 → 0.04), so probabilities tighten up too.

## Reproduce

```bash
python -m src.train --skip-tabpfn
```

Writes `outputs/metrics.json`. Committed snapshot: `outputs/metrics.example.json`.

## Add another model

Drop a `train_<name>()` in `src/train.py`, register it in `benchmarks` and `_print_table`, re-run, update the example json.
