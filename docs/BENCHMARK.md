# Benchmark

Same feature matrix and holdout for every model. The question I cared about: after hand-engineering types and stats, how much does the classifier still matter?

## Split

| | |
|---|---|
| Labeled rows | 50,000 (`Test_Set=0`) |
| Train | 42,500 |
| Holdout | 7,500 (15%, stratified, seed 42) |
| Excluded | 2,080 `Test_Set=1` rows (Kaggle export with no winner label) |

## Models

**Logistic regression** — standardized features, sklearn LR. Sanity check for linear separability.

**CatBoost** — 300 iterations, depth 8, learning rate 0.08, early stopping on holdout AUC. Saved to `models/catboost.cbm`.

**TabPFN** — optional Prior Labs comparator. Fits on up to 10,000 stratified train rows, evaluates on the same 7,500 holdout. Needs `TABPFN_TOKEN` and a one-time license at [ux.priorlabs.ai](https://ux.priorlabs.ai). CPU runs need `TABPFN_ALLOW_CPU_LARGE_DATASET=1` (set in `.env.example`; `train.py` applies it).

## Results (June 2026)

| Model | Accuracy | ROC-AUC | Log loss |
|-------|----------|---------|----------|
| Logistic regression | 0.9185 | 0.9605 | 0.2338 |
| CatBoost | 0.9852 | 0.9990 | 0.0441 |
| TabPFN | 0.9760 | 0.9977 | 0.0616 |

TabPFN on CPU took about six minutes for the 10k fit + holdout score.

Log loss drops from 0.23 (logistic) to 0.04 (CatBoost), so probabilities tighten as well as hard accuracy.

## Reproduce

```bash
python -m src.train
```

Writes `outputs/metrics.json`. Checked-in snapshot: `outputs/metrics.example.json`.
