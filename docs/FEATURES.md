# Features (70 columns)

One row per combat. Label: `First_Winner` (1 if the first pokemon id won).

Column order: `src.features.feature_column_names()`.

## Pipeline

```text
pokemon.csv + chart.csv + species.csv
  → per-mon stats (MCA types, raw stats, legendary/mega/stage)
  → pairwise row (deltas, type chart, matchup fields)
  → 70 floats
```

## Per-mon side (`_x` first, `_y` second)

| Block | Count | Source |
|-------|-------|--------|
| MCA types | 15 each | `prince` MCA on `(Type_1, Type_2)`; carried over from 2018 |
| Base stats | 6 each | HP, Attack, Defense, Sp_Atk, Sp_Def, Speed |
| Meta | 4 each | Legendary, mega, baby, evolution stage |

Computed internally but not exported as columns: BST, Offense, Bulk, Mono_Type.

## Stat deltas

`HP_Delta` … `Speed_Delta` (first minus second). From the original notebook.

## Type chart

| Column | Meaning |
|--------|---------|
| `First_Attacker_Eff` | Multiplier for first mon's type combo vs second's types |
| `Second_Attacker_Eff` | Reverse |

See [CONTEXT.md](CONTEXT.md) for how dual-type multiplication works.

## Matchup fields (2026 additions)

| Column | Meaning |
|--------|---------|
| `BST_Delta` | Base stat total gap |
| `Offense_Delta` | Atk+SpA gap |
| `Bulk_Delta` | HP+Def+SpD gap |
| `Stage_Delta` | Evolution stage gap |
| `Legendary_Gap` | Legendary flag difference |
| `Mono_Type_x/y` | 1 if single-typed |
| `Type_Eff_Delta` | First minus second attacker effectiveness |
| `Type_Eff_LogRatio` | log ratio of effectiveness values |
| `Speed_Ratio` | Speed share of combined speed |
| `First_Speed` | 1 if first is faster or tied |
| `Shared_Type` | 1 if any type overlaps |

## Omitted

- Raw type one-hots (MCA instead)
- Names / ids in X
- Moves, abilities, items (not in combat CSV)

## Code entry points

| Function | Use |
|----------|-----|
| `get_enriched_stats()` | Cached per-mon table |
| `build_combat_frame()` | Full labeled matrix |
| `build_matchup_features(a, b)` | Single row for predict; must match train columns |

After edits: `python -m src.train` and refresh `outputs/metrics.example.json`.
