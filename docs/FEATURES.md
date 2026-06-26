# Feature engineering

Each combat row is a **pair** of Pokémon (first vs second). The label is `First_Winner`: 1 if the first id in the combat file won.

Total training columns: **70** (see `src.features.feature_column_names()`).

## Per-Pokémon blocks (first `_x`, second `_y`)

| Block | Columns | Notes |
|-------|---------|--------|
| MCA types | 15 × 2 | `prince` MCA on `(Type_1, Type_2)`, same as the 2018 script |
| Base stats | HP, Attack, Defense, Sp_Atk, Sp_Def, Speed × 2 | Raw from `pokemon.csv` |
| Meta | is_Legendary, is_Mega, is_baby, Stage × 2 | Legendary flag + evolution chain pass |

Internal per-mon derivations (used to build matchup features, not exported as `_x/_y` columns): **BST** (sum of six stats), **Offense** (Attack + Sp_Atk), **Bulk** (HP + Defense + Sp_Def), **Mono_Type** (1 if single-typed).

## Stat deltas (first minus second)

`HP_Delta`, `Attack_Delta`, `Defense_Delta`, `Sp_Atk_Delta`, `Sp_Def_Delta`, `Speed_Delta`

Direct carry-over from the original notebook.

## Type chart (from `chart.csv`)

| Column | Meaning |
|--------|---------|
| `First_Attacker_Eff` | Multiplier when first Pokémon’s combined type attacks second’s types |
| `Second_Attacker_Eff` | Reverse direction |

Combined type string is `Type_1 + Type_2` (e.g. `GrassPoison`). Chart includes pairwise type-product columns from the 2018 expansion.

## Matchup derivations (added 2026)

| Column | Meaning |
|--------|---------|
| `BST_Delta` | Base-stat-total advantage |
| `Offense_Delta` | (Attack + Sp_Atk) advantage |
| `Bulk_Delta` | (HP + Defense + Sp_Def) advantage |
| `Stage_Delta` | Evolution stage difference |
| `Legendary_Gap` | `is_Legendary_x − is_Legendary_y` |
| `Mono_Type_x`, `Mono_Type_y` | Single-type flags |
| `Type_Eff_Delta` | `First_Attacker_Eff − Second_Attacker_Eff` |
| `Type_Eff_LogRatio` | log ratio of the two effectiveness values |
| `Speed_Ratio` | `Speed_x / (Speed_x + Speed_y)` |
| `First_Speed` | 1 if first Pokémon is faster or tied |
| `Shared_Type` | 1 if any non-None type overlaps |

## What we deliberately omit

- Raw type one-hots (MCA covers types more compactly)
- Pokémon names or ids in the feature matrix (leakage / memorization)
- `Test_Set=1` unlabeled Kaggle rows in training or metrics

## Code map

| Function | Role |
|----------|------|
| `get_enriched_stats()` | Cached per-Pokémon table |
| `build_combat_frame()` | Full labeled matrix for training |
| `build_matchup_features(a, b)` | One row for `predict.py`; column order must match training |

After changing features, re-run `python -m src.train` and update `outputs/metrics.example.json`.
