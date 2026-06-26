# Dataset and game context

Notes on what the Kaggle battles actually represent and what we left out.

## Where the battles come from

The combat CSVs are from the public **Predict Pokemon Battles** Kaggle task (circa 2018). Each labeled row is a simulated 1v1 between two species drawn from `pokemon.csv`. The winner column is whichever side the simulator recorded.

Important limitation: **we do not get movesets, held items, abilities, or levels.** Only species ids and who won. That is why this project leans on base stats and type matchups instead of move pools or damage calc.

The 2,080 `Test_Set=1` rows are the Kaggle holdout export with no winner label. This repo ignores them for training and metrics.

## Type chart (what `chart.csv` encodes)

Mainline games multiply effectiveness across both defending types. Water vs Charizard (Fire/Flying): Fire takes 2×, Flying takes 1×, so the combined multiplier is 2×.

The 2018 pipeline concatenates attacking types (`Grass` + `Poison` → `GrassPoison`) and looks up multipliers against each defending type, then multiplies. Dual-type columns in `chart.csv` are the pairwise products from that expansion.

We export two directions per fight:

- `First_Attacker_Eff`: first mon's combined type attacking second's types
- `Second_Attacker_Eff`: the reverse

`Type_Eff_Delta` and `Type_Eff_LogRatio` compress that matchup into one number each.

## Stats and competitive shorthand

`pokemon.csv` uses the usual six base stats (HP, Atk, Def, SpA, SpD, Spe). Megas are separate rows with inflated totals. Legendaries skew high on BST.

Features borrowed from Smogon-style thinking without pretending this is VGC:

| Derived | Definition | Why it shows up in fights |
|---------|------------|---------------------------|
| BST | Sum of six stats | Raw budget; correlates with bulk and power |
| Offense | Atk + SpA | Physical/special pressure combined |
| Bulk | HP + Def + SpD | Tolerance before KO |
| Speed ratio | Spe_x / (Spe_x + Spe_y) | Turn order proxy; higher Speed acts first in-game |
| Stage | Evolution step in chain | Later stages usually have higher stats |

Speed does not appear in the CSV winner logic explicitly, but faster species win more often in the simulations, so `First_Speed` and `Speed_Ratio` help.

## What we tried not to leak

- **Pokemon names and ids** stay out of the feature matrix. The model should generalize from stats/types, not memorize "Mewtwo always wins."
- **Test_Set=1** unlabeled rows never enter fit or metrics.

## Movesets (not in this dataset)

Real battles depend heavily on moves (coverage, priority, status). `pokedex.csv` lists abilities and per-type damage multipliers (`against_fire`, etc.) but the combat files never say which four moves each side brought.

Possible extensions if you merge external data:

- Level-up / TM move lists per species → count of super-effective moves vs opponent typing
- Ability flags from `pokedex.csv` (e.g. Levitate, Thick Fat)
- `against_*` columns as a second type-resistance vector

None of that is in the current 70-column matrix; documenting it here so the scope is clear.

## Id column vs national dex

`pokemon.csv` `#` is the row index in this file (1–721 with gaps for formes). It is **not** the national dex number. `predict.py --a 163 --b 7` uses file ids (Mewtwo vs Charizard in this export).

## References

- Kaggle: [Predict Pokemon Battles](https://www.kaggle.com/competitions/pokemon-challenge) (archived task)
- Type chart mechanics: [Bulbapedia – Type](https://bulbapedia.bulbagarden.net/wiki/Type)
- Original 2018 code in `legacy/Pokemon_Battle_Match.py`
