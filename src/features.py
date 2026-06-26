"""Feature engineering for 1v1 battle outcome prediction."""

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


def _load_stats() -> pd.DataFrame:
    stats = pd.read_csv(DATA_DIR / "pokemon.csv")
    stats.columns = [c.replace(" ", "_").replace(".", "") for c in stats.columns]
    stats[["Type_1", "Type_2"]] = stats[["Type_1", "Type_2"]].fillna("None")
    stats.loc[stats.index[62], "Name"] = "Primeape"
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
            if row["is_baby"] == 1:
                chain_stage = 0
            else:
                chain_stage += 1
        stage.append(chain_stage)
    merged["Stage"] = stage
    return merged.set_index("#")[["is_Mega", "is_baby", "Stage"]]


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
            name = f"{left}{right}"
            frames.append((transposed[left] * transposed[right]).rename(name))
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


@lru_cache(maxsize=1)
def get_enriched_stats() -> pd.DataFrame:
    stats = _load_stats()
    mca = _mca_type_components(stats)
    evo = _evolution_features(stats)
    return pd.concat([mca.set_index(stats["#"]), stats.set_index("#"), evo], axis=1)


def pokemon_name(pokemon_id: int) -> str:
    row = get_enriched_stats().loc[pokemon_id]
    return str(row["Name"]).title()


def build_matchup_features(first_id: int, second_id: int) -> pd.DataFrame:
    """Single-row feature matrix aligned with training columns."""
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

    mtype_a = str(a["MType"])
    mtype_b = str(b["MType"])

    def eff(att_mtype: str, d1: str, d2: str) -> float:
        e1 = matrix.at[d1, att_mtype] if d1 in matrix.index and att_mtype in matrix.columns else 1.0
        e2 = matrix.at[d2, att_mtype] if d2 in matrix.index and att_mtype in matrix.columns else 1.0
        return float(e1 * e2)

    row["First_Attacker_Eff"] = eff(mtype_a, str(b["Type_1"]), str(b["Type_2"]))
    row["Second_Attacker_Eff"] = eff(mtype_b, str(a["Type_1"]), str(a["Type_2"]))
    row["First_Speed"] = float(a["Speed"] >= b["Speed"])
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
    df["First_Speed"] = (df["Speed_Delta"] >= 0).astype(int)

    keep = (
        ["First_Winner", "Test_Set"]
        + [f"{c}_x" for c in PAIR_BASE_COLS]
        + [f"{c}_y" for c in PAIR_BASE_COLS]
        + [f"{c}_Delta" for c in STAT_COLS]
        + ["First_Attacker_Eff", "Second_Attacker_Eff", "First_Speed"]
    )
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
