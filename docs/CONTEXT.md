# Dataset notes

Background on the Kaggle battles export and what the model can and cannot see.

## What is in the combat file

The **Predict Pokémon Battles** Kaggle task (2018) provides simulated 1v1 matchups between species in `pokemon.csv`. Each labeled row records two ids and which side won.

There are no movesets, held items, abilities, or levels in the combat CSV. The pipeline uses base stats, typings, evolution stage, and the type chart because that is what the data supports.

2,080 rows flagged `Test_Set=1` have no winner (Kaggle submission holdout). Training and metrics use only the 50,000 labeled rows.

## Type chart

Mainline games multiply effectiveness across both defending types. Example: Water into Charizard (Fire/Flying) hits Fire at 2× and Flying at 1×, so the combined multiplier is 2×.

The 2018 approach concatenates attack typings (`Grass` + `Poison` → `GrassPoison`) and multiplies lookups against each defending type. `chart.csv` includes expanded dual-type columns from that step.

Each fight gets two directional features: `First_Attacker_Eff` and `Second_Attacker_Eff`, plus deltas derived from them.

## Stats

Six base stats per species. Megas are separate rows with higher totals. Legendaries tend to have larger BST.

| Derived | Definition | Why it helps |
|---------|------------|--------------|
| BST | Sum of six stats | Overall stat budget |
| Offense | Atk + SpA | Combined attacking pressure |
| Bulk | HP + Def + SpD | Staying power |
| Speed ratio | Spe_x / (Spe_x + Spe_y) | Turn-order proxy |
| Stage | Step in evolution chain | Later stages usually stronger |

Faster species win more often in the simulations, so speed features matter even though turn order is not explicit in the CSV.

## Leakage choices

Species names and ids never enter the feature matrix. The model should learn from stats and types, not memorize that a particular id always wins.

## Movesets (future work)

Real battles hinge on move coverage, priority, and status. `pokedex.csv` has abilities and `against_*` resistances, but combats do not list moves.

Reasonable extensions if you pull external move data:

- Count of super-effective moves vs opponent typing
- Ability flags from `pokedex.csv`
- Resistance vector from `against_*` columns

None of that is in the current 70-column set.

## Id column

`pokemon.csv` `#` is the row index in this file, not the national dex. Example: `python -m src.predict --a 163 --b 7` is Mewtwo vs Charizard in this export.

## References

- [Bulbapedia: Type](https://bulbapedia.bulbagarden.net/wiki/Type)
- Original pipeline: `legacy/Pokemon_Battle_Match.py`
