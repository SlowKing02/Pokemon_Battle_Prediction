"""Feature engineering for 1v1 battle outcome prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import prince

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

STAT_COLS = ["HP", "Attack", "Defense", "Sp_Atk", "Sp_Def", "Speed"]


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
    components.columns = [f"mca_{i}" for i in range(components.shape[1])]
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

    stage = []
    prev_chain: int | None = None
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


def _build_type_effect_matrix() -> pd.DataFrame:
    """Expanded type chart: columns include single types and type-pair combos."""
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


def _attack_effect(
    attacker_mtype: pd.Series,
    def_type1: pd.Series,
    def_type2: pd.Series,
    matrix: pd.DataFrame,
) -> pd.Series:
    eff1 = matrix.lookup(def_type1, attacker_mtype)
    eff2 = matrix.lookup(def_type2, attacker_mtype)
    return eff1 * eff2


def _attack_effect_vectorized(
    attacker_mtype: pd.Series,
    def_type1: pd.Series,
    def_type2: pd.Series,
    matrix: pd.DataFrame,
) -> np.ndarray:
    """Vectorized type effectiveness (replaces 52k-row Python loops)."""
    m = matrix.to_numpy()
    col_index = {c: i for i, c in enumerate(matrix.columns)}
    row_index = {r: i for i, r in enumerate(matrix.index)}

    def lookup(series_row: pd.Series, series_col: pd.Series) -> np.ndarray:
        out = np.ones(len(series_row), dtype=float)
        for i, (r, c) in enumerate(zip(series_row, series_col)):
            ri = row_index.get(r)
            ci = col_index.get(c)
            if ri is not None and ci is not None:
                out[i] = m[ri, ci]
            elif c in col_index:
                out[i] = 1.0
        return out

    # Faster path: use .reindex + numpy indexing where keys exist
    col_idx = attacker_mtype.map(col_index).to_numpy()
    row1_idx = def_type1.map(row_index).to_numpy()
    row2_idx = def_type2.map(row_index).to_numpy()

    valid = (col_idx >= 0) & (row1_idx >= 0) & (row2_idx >= 0)
    eff = np.ones(len(attacker_mtype), dtype=float)
    eff[valid] = m[row1_idx[valid], col_idx[valid]] * m[row2_idx[valid], col_idx[valid]]
    return eff


def build_combat_frame() -> pd.DataFrame:
    """Full labeled combat frame with engineered features."""
    combats = pd.read_csv(DATA_DIR / "combats_test.csv")
    stats = _load_stats()
    mca = _mca_type_components(stats)
    evo = _evolution_features(stats)
    effect_matrix = _build_type_effect_matrix()

    base = stats.set_index("#")
    enriched = pd.concat([mca.set_index(stats["#"]), base, evo], axis=1)

    feature_cols = (
        list(mca.columns)
        + STAT_COLS
        + ["is_Legendary", "is_Mega", "is_baby", "Stage"]
    )

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
        + [f"{c}_x" for c in feature_cols]
        + [f"{c}_y" for c in feature_cols]
        + [f"{c}_Delta" for c in STAT_COLS]
        + ["First_Attacker_Eff", "Second_Attacker_Eff", "First_Speed"]
    )
    return df[keep].copy()


def labeled_combat_frame() -> pd.DataFrame:
    """Rows with known winners (Test_Set=0). Test_Set=1 is unlabeled Kaggle holdout."""
    return build_combat_frame().query("Test_Set == 0").drop(columns=["Test_Set"]).copy()


def train_test_split_holdout(
    frame: pd.DataFrame,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Stratified holdout from labeled combats."""
    from sklearn.model_selection import train_test_split

    x = frame.drop(columns=["First_Winner"])
    y = frame["First_Winner"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return x_train, y_train, x_test, y_test


def unlabeled_submission_frame() -> pd.DataFrame:
    """Kaggle-style combats without winner labels (Test_Set=1)."""
    return build_combat_frame().query("Test_Set == 1").drop(columns=["Test_Set", "First_Winner"])
