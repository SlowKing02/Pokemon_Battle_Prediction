# Build the combat feature matrix from pokemon.csv, chart.csv, species.csv.
# Started from my 2018 grad-school script; 2026 pass added matchup deltas.

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import prince

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

STAT_COLS = ["HP", "Attack", "Defense", "Sp_Atk", "Sp_Def", "Speed"]
MCA_COLS = [f"mca_{i}" for i in range(15)]
META_COLS = ["is_Legendary", "is_Mega", "is_baby", "Stage"]
PAIR_BASE_COLS = MCA_COLS + STAT_COLS + META_COLS

MATCHUP_DERIVED_COLS = [
    "BST_Delta",
    "Offense_Delta",
    "Bulk_Delta",
    "Stage_Delta",
    "Legendary_Gap",
    "Mono_Type_x",
    "Mono_Type_y",
    "Type_Eff_Delta",
    "Type_Eff_LogRatio",
    "Speed_Ratio",
    "First_Speed",
    "Shared_Type",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "mca_types_x": [f"{c}_x" for c in MCA_COLS],
    "mca_types_y": [f"{c}_y" for c in MCA_COLS],
    "base_stats_x": [f"{c}_x" for c in STAT_COLS],
    "base_stats_y": [f"{c}_y" for c in STAT_COLS],
    "meta_x": [f"{c}_x" for c in META_COLS],
    "meta_y": [f"{c}_y" for c in META_COLS],
    "stat_deltas": [f"{c}_Delta" for c in STAT_COLS],
    "type_chart": ["First_Attacker_Eff", "Second_Attacker_Eff"],
    "matchup_derived": MATCHUP_DERIVED_COLS,
}


def feature_column_names() -> list[str]:
    cols: list[str] = []
    for group in FEATURE_GROUPS:
        cols.extend(FEATURE_GROUPS[group])
    return cols


def _load_stats() -> pd.DataFrame:
    stats = pd.read_csv(DATA_DIR / "pokemon.csv")
    stats.columns = [c.replace(" ", "_").replace(".", "") for c in stats.columns]
    stats[["Type_1", "Type_2"]] = stats[["Type_1", "Type_2"]].fillna("None")
    stats.loc[stats.index[62], "Name"] = "Primeape"  # typo in the Kaggle export
    stats["Name"] = stats["Name"].str.lower()
    stats["MType"] = stats["Type_1"] + stats["Type_2"]
    stats["is_Legendary"] = stats["Legendary"].astype(int)
    return stats


def _mca_type_components(stats: pd.DataFrame, n_components: int = 15) -> pd.DataFrame:
    types = stats[["Type_1", "Type_2"]].fillna("None")
    mca = prince.MCA(
        n_components=n_components,
        n_iter=100,
        copy=True,
        engine="sklearn",
        random_state=42,
    )
    mca.fit(types)
    components = pd.DataFrame(mca.transform(types), index=stats.index)
    components.columns = MCA_COLS
    return components


def _evolution_features(stats: pd.DataFrame) -> pd.DataFrame:
    evolution = pd.read_csv(DATA_DIR / "pokemon_species.csv")
    merged = stats[["Name", "#"]].merge(
        evolution[["identifier", "evolution_chain_id", "is_baby"]],
        left_on="Name",
        right_on="identifier",
        how="left",
    )
    merged["is_Mega"] = merged["Name"].str.startswith("Mega ").astype(int)
    merged["is_baby"] = merged["is_baby"].fillna(0).astype(int)
    merged = merged.sort_values(["evolution_chain_id", "#"]).reset_index(drop=True)

    stage: list[int] = []
    prev_chain: float | None = None
    chain_stage = 0
    for _, row in merged.iterrows():
        chain = row["evolution_chain_id"]
        if chain != prev_chain:
            chain_stage = 1 if row["is_baby"] == 0 else 0
            prev_chain = chain
        else:
            chain_stage = 0 if row["is_baby"] == 1 else chain_stage + 1
        stage.append(chain_stage)
    merged["Stage"] = stage
    return merged.set_index("#")[["is_Mega", "is_baby", "Stage"]]


def _attach_per_mon_derivations(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["BST"] = out[STAT_COLS].sum(axis=1)
    out["Offense"] = out["Attack"] + out["Sp_Atk"]
    out["Bulk"] = out["HP"] + out["Defense"] + out["Sp_Def"]
    out["Mono_Type"] = (out["Type_2"] == "None").astype(int)
    return out


@lru_cache(maxsize=1)
def _build_type_effect_matrix() -> pd.DataFrame:
    chart = pd.read_csv(DATA_DIR / "chart.csv")
    chart = chart.set_index("Attacking")
    chart["None"] = 1.0
    transposed = chart.T
    transposed["None"] = 1.0

    frames = [transposed]
    cols = list(transposed.columns)
    for left in cols:
        for right in cols:
            frames.append((transposed[left] * transposed[right]).rename(f"{left}{right}"))
    return pd.concat(frames, axis=1)


def _attack_effect_vectorized(
    attacker_mtype: pd.Series,
    def_type1: pd.Series,
    def_type2: pd.Series,
    matrix: pd.DataFrame,
) -> np.ndarray:
    m = matrix.to_numpy()
    col_index = {c: i for i, c in enumerate(matrix.columns)}
    row_index = {r: i for i, r in enumerate(matrix.index)}

    col_idx = attacker_mtype.map(col_index).fillna(-1).astype(int).to_numpy()
    row1_idx = def_type1.map(row_index).fillna(-1).astype(int).to_numpy()
    row2_idx = def_type2.map(row_index).fillna(-1).astype(int).to_numpy()

    valid = (col_idx >= 0) & (row1_idx >= 0) & (row2_idx >= 0)
    eff = np.ones(len(attacker_mtype), dtype=float)
    eff[valid] = m[row1_idx[valid], col_idx[valid]] * m[row2_idx[valid], col_idx[valid]]
    return eff


def _type_effect(att_mtype: str, d1: str, d2: str, matrix: pd.DataFrame) -> float:
    e1 = matrix.at[d1, att_mtype] if d1 in matrix.index and att_mtype in matrix.columns else 1.0
    e2 = matrix.at[d2, att_mtype] if d2 in matrix.index and att_mtype in matrix.columns else 1.0
    return float(e1 * e2)


def _shared_type_flag(type_1_a: str, type_2_a: str, type_1_b: str, type_2_b: str) -> float:
    ta = {type_1_a, type_2_a} - {"None"}
    tb = {type_1_b, type_2_b} - {"None"}
    return float(len(ta & tb) > 0)


def _derived_matchup_row(
    a: pd.Series,
    b: pd.Series,
    first_eff: float,
    second_eff: float,
) -> dict[str, float]:
    speed_sum = float(a["Speed"] + b["Speed"]) + 1e-6
    return {
        "BST_Delta": float(a["BST"] - b["BST"]),
        "Offense_Delta": float(a["Offense"] - b["Offense"]),
        "Bulk_Delta": float(a["Bulk"] - b["Bulk"]),
        "Stage_Delta": float(a["Stage"] - b["Stage"]),
        "Legendary_Gap": float(a["is_Legendary"] - b["is_Legendary"]),
        "Mono_Type_x": float(a["Mono_Type"]),
        "Mono_Type_y": float(b["Mono_Type"]),
        "Type_Eff_Delta": float(first_eff - second_eff),
        "Type_Eff_LogRatio": float(np.log((first_eff + 1e-6) / (second_eff + 1e-6))),
        "Speed_Ratio": float(a["Speed"] / speed_sum),
        "First_Speed": float(a["Speed"] >= b["Speed"]),
        "Shared_Type": _shared_type_flag(
            str(a["Type_1"]), str(a["Type_2"]), str(b["Type_1"]), str(b["Type_2"])
        ),
    }


def _append_matchup_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["BST_Delta"] = out["BST_x"] - out["BST_y"]
    out["Offense_Delta"] = out["Offense_x"] - out["Offense_y"]
    out["Bulk_Delta"] = out["Bulk_x"] - out["Bulk_y"]
    out["Stage_Delta"] = out["Stage_x"] - out["Stage_y"]
    out["Legendary_Gap"] = out["is_Legendary_x"] - out["is_Legendary_y"]
    out["Type_Eff_Delta"] = out["First_Attacker_Eff"] - out["Second_Attacker_Eff"]
    out["Type_Eff_LogRatio"] = np.log(
        (out["First_Attacker_Eff"] + 1e-6) / (out["Second_Attacker_Eff"] + 1e-6)
    )
    speed_sum = out["Speed_x"] + out["Speed_y"] + 1e-6
    out["Speed_Ratio"] = out["Speed_x"] / speed_sum
    out["First_Speed"] = (out["Speed_Delta"] >= 0).astype(int)
    out["Shared_Type"] = [
        _shared_type_flag(r.Type_1_x, r.Type_2_x, r.Type_1_y, r.Type_2_y)  # type: ignore[attr-defined]
        for r in out.itertuples(index=False)
    ]
    return out


@lru_cache(maxsize=1)
def get_enriched_stats() -> pd.DataFrame:
    stats = _load_stats()
    mca = _mca_type_components(stats)
    evo = _evolution_features(stats)
    base = pd.concat([mca.set_index(stats["#"]), stats.set_index("#"), evo], axis=1)
    return _attach_per_mon_derivations(base)


def pokemon_name(pokemon_id: int) -> str:
    return str(get_enriched_stats().loc[pokemon_id]["Name"]).title()


def build_matchup_features(first_id: int, second_id: int) -> pd.DataFrame:
    enriched = get_enriched_stats()
    if first_id not in enriched.index or second_id not in enriched.index:
        missing = [i for i in (first_id, second_id) if i not in enriched.index]
        raise ValueError(f"Unknown pokemon id(s): {missing}")

    a = enriched.loc[first_id]
    b = enriched.loc[second_id]
    matrix = _build_type_effect_matrix()

    row: dict[str, float] = {}
    for col in PAIR_BASE_COLS:
        row[f"{col}_x"] = float(a[col])
        row[f"{col}_y"] = float(b[col])
    for stat in STAT_COLS:
        row[f"{stat}_Delta"] = float(a[stat] - b[stat])

    first_eff = _type_effect(str(a["MType"]), str(b["Type_1"]), str(b["Type_2"]), matrix)
    second_eff = _type_effect(str(b["MType"]), str(a["Type_1"]), str(a["Type_2"]), matrix)
    row["First_Attacker_Eff"] = first_eff
    row["Second_Attacker_Eff"] = second_eff
    row.update(_derived_matchup_row(a, b, first_eff, second_eff))
    return pd.DataFrame([row])


def build_combat_frame() -> pd.DataFrame:
    combats = pd.read_csv(DATA_DIR / "combats_test.csv")
    enriched = get_enriched_stats()
    effect_matrix = _build_type_effect_matrix()

    first = enriched.add_suffix("_x")
    second = enriched.add_suffix("_y")
    df = combats.merge(first, left_on="First_pokemon", right_index=True, how="inner")
    df = df.merge(second, left_on="Second_pokemon", right_index=True, how="inner")

    df["First_Winner"] = (df["First_pokemon"] == df["Winner"]).astype(int)
    df["First_Attacker_Eff"] = _attack_effect_vectorized(
        df["MType_x"], df["Type_1_y"], df["Type_2_y"], effect_matrix
    )
    df["Second_Attacker_Eff"] = _attack_effect_vectorized(
        df["MType_y"], df["Type_1_x"], df["Type_2_x"], effect_matrix
    )

    for stat in STAT_COLS:
        df[f"{stat}_Delta"] = df[f"{stat}_x"] - df[f"{stat}_y"]

    df = _append_matchup_derived(df)
    keep = ["First_Winner", "Test_Set"] + feature_column_names()
    return df[keep].copy()


def labeled_combat_frame() -> pd.DataFrame:
    return build_combat_frame().query("Test_Set == 0").drop(columns=["Test_Set"]).copy()


def train_test_split_holdout(
    frame: pd.DataFrame,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    from sklearn.model_selection import train_test_split

    x = frame.drop(columns=["First_Winner"])
    y = frame["First_Winner"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return x_train, y_train, x_test, y_test
