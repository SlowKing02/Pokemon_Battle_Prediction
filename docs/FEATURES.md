# Features

Each row is one combat: two Pokémon (first vs second). Label: `First_Winner` (1 if the first id won).

**70 columns** total. List from `src.features.feature_column_names()`.

## Per-Pokémon blocks (`_x` = first, `_y` = second)

| Block | Columns | Notes |
|-------|---------|--------|
| MCA types | 15 × 2 | `prince` MCA on `(Type_1, Type_2)`; same approach as the 2018 script |
| Base stats | HP, Attack, Defense, Sp_Atk, Sp_Def, Speed × 2 | From `pokemon.csv` |
| Meta | is_Legendary, is_Mega, is_baby, Stage × 2 | Legendary flag + evolution chain |

Per-mon derivations used internally (not exported as `_x/_y` columns): **BST** (sum of six stats), **Offense** (Attack + Sp_Atk), **Bulk** (HP + Defense + Sp_Def), **Mono_Type** (1 if single-typed).

## Stat deltas (first minus second)

`HP_Delta`, `Attack_Delta`, `Defense_Delta`, `Sp_Atk_Delta`, `Sp_Def_Delta`, `Speed_Delta`

From the original notebook.

## Type chart (`chart.csv`)

| Column | Meaning |
|--------|---------|
| `First_Attacker_Eff` | Multiplier when first Pokémon's combined type attacks second's types |
| `Second_Attacker_Eff` | Reverse direction |

Combined type string is `Type_1 + Type_2` (e.g. `GrassPoison`). Chart includes pairwise type-product columns from the 2018 expansion.

## Matchup derivations (2026)

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

## Omitted on purpose

- Raw type one-hots (MCA is enough)
- Pokémon names or ids in the matrix (memorization)
- `Test_Set=1` unlabeled Kaggle rows

## Code

| Function | Role |
|----------|------|
| `get_enriched_stats()` | Cached per-Pokémon table |
| `build_combat_frame()` | Full labeled matrix for training |
| `build_matchup_features(a, b)` | One row for `predict.py`; must match training columns |

After changing features: `python -m src.train` and update `outputs/metrics.example.json`.
