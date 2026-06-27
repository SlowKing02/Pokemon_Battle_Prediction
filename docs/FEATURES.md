# Features

70 columns per combat row. Label: `First_Winner` (1 if the first species id won).

Order: `src.features.feature_column_names()`.

## How rows are built

```text
pokemon.csv + chart.csv + species.csv
  → per-species table (MCA types, stats, legendary/mega/stage)
  → pairwise row (deltas, type chart, matchup fields)
```

## Per-species side (`_x` = first, `_y` = second)

| Block | Count | Notes |
|-------|-------|-------|
| MCA types | 15 each | `prince` MCA on `(Type_1, Type_2)` from the 2018 script |
| Base stats | 6 each | HP, Attack, Defense, Sp_Atk, Sp_Def, Speed |
| Meta | 4 each | Legendary, mega, baby, evolution stage |

Internal only (not exported as `_x/_y`): BST, Offense (Atk+SpA), Bulk (HP+Def+SpD), Mono_Type.

## Stat deltas

Six columns, first minus second: `HP_Delta` through `Speed_Delta`.

## Type chart

| Column | Meaning |
|--------|---------|
| `First_Attacker_Eff` | Multiplier for first mon's type combo vs second's types |
| `Second_Attacker_Eff` | Reverse direction |

Dual-type math is in [CONTEXT.md](CONTEXT.md).

## Matchup fields (2026)

| Column | Meaning |
|--------|---------|
| `BST_Delta` | Base stat total gap |
| `Offense_Delta` | Atk+SpA gap |
| `Bulk_Delta` | HP+Def+SpD gap |
| `Stage_Delta` | Evolution stage gap |
| `Legendary_Gap` | Legendary flag difference |
| `Mono_Type_x/y` | Single-type indicator |
| `Type_Eff_Delta` | First minus second attacker effectiveness |
| `Type_Eff_LogRatio` | Log ratio of effectiveness values |
| `Speed_Ratio` | Speed share of combined speed |
| `First_Speed` | 1 if first is faster or tied |
| `Shared_Type` | 1 if any type overlaps |

## Left out on purpose

- Pokémon names and ids in X (memorization, not generalization)
- Raw type one-hots (MCA covers typings)
- Moves, abilities, items (not in the combat CSV)

## Code

| Function | Role |
|----------|------|
| `get_enriched_stats()` | Cached per-species table |
| `build_combat_frame()` | Full labeled matrix for training |
| `build_matchup_features(a, b)` | One row for `predict.py` |

After changing columns, re-run `python -m src.train` and update `outputs/metrics.example.json`.
