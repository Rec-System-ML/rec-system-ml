from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler


@dataclass
class KuaiRandTables:
    interactions: pd.DataFrame
    users: pd.DataFrame
    items: pd.DataFrame


def _resolve_existing_file(base: Path, preferred: str, fallbacks: Sequence[str]) -> Optional[Path]:
    preferred_path = base / preferred
    if preferred_path.exists():
        return preferred_path
    for name in fallbacks:
        path = base / name
        if path.exists():
            return path
    return None


def parse_tag_ids(value: object) -> List[int]:
    """Parse tag field into a list of integer tag ids."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value if str(v).strip() != ""]

    text = str(value).strip()
    if not text:
        return []

    tags: List[int] = []
    for chunk in text.replace(";", ",").split(","):
        token = chunk.strip()
        if not token:
            continue
        try:
            tags.append(int(token))
        except ValueError:
            # Ignore malformed token, keep pipeline robust.
            continue
    return tags


def _pick_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_kuairand_tables(
    data_path: str | Path,
    interactions_file: str = "interactions.csv",
    users_file: str = "user_features.csv",
    items_file: str = "video_features.csv",
) -> KuaiRandTables:
    """Load core KuaiRand tables from csv files."""
    base = Path(data_path)
    if not base.exists():
        raise FileNotFoundError(f"Data path does not exist: {base}")
    # Convenience: allow passing repo root that contains data/KuaiRand-1K/data.
    nested_kuairand = base / "data" / "KuaiRand-1K" / "data"
    if nested_kuairand.exists():
        base = nested_kuairand

    interactions_path = base / interactions_file
    if interactions_path.exists():
        interactions = pd.read_csv(interactions_path)
    else:
        # KuaiRand-1K often ships split logs. Prefer standard logs first.
        standard_logs = sorted(base.glob("log_standard*_1k.csv"))
        random_logs = sorted(base.glob("log_random*_1k.csv"))
        candidate_logs = standard_logs or random_logs
        if not candidate_logs:
            raise FileNotFoundError(
                f"Interactions file not found: {interactions_path}. "
                "Also failed to find KuaiRand logs like log_standard*_1k.csv"
            )
        interactions = pd.concat([pd.read_csv(p) for p in candidate_logs], ignore_index=True)

    users_path = _resolve_existing_file(
        base,
        preferred=users_file,
        fallbacks=["user_features_1k.csv", "user_feature_1k.csv"],
    )
    if users_path is None:
        raise FileNotFoundError(f"Users file not found under: {base}")
    users = pd.read_csv(users_path)

    items_path = _resolve_existing_file(
        base,
        preferred=items_file,
        fallbacks=["video_features_basic_1k.csv", "video_features_basic.csv", "video_features_statistic_1k.csv"],
    )
    if items_path is None:
        raise FileNotFoundError(f"Items file not found under: {base}")
    items = pd.read_csv(items_path)
    return KuaiRandTables(interactions=interactions, users=users, items=items)


def load_data(
    data_path: str | Path,
    interactions_file: str = "interactions.csv",
    users_file: str = "user_features.csv",
    items_file: str = "video_features.csv",
) -> KuaiRandTables:
    """Backward-compatible alias used by both CDS524/CDS525 demos."""
    return load_kuairand_tables(
        data_path=data_path,
        interactions_file=interactions_file,
        users_file=users_file,
        items_file=items_file,
    )


def _stratified_downsample(
    df: pd.DataFrame, label_col: str, sample_size: int, random_state: int
) -> pd.DataFrame:
    if sample_size >= len(df):
        return df.copy()

    frac = sample_size / len(df)
    grouped = (
        df.groupby(label_col, dropna=False, group_keys=False)
        .apply(lambda x: x.sample(max(1, int(round(len(x) * frac))), random_state=random_state))
        .reset_index(drop=True)
    )
    if len(grouped) > sample_size:
        grouped = grouped.sample(sample_size, random_state=random_state).reset_index(drop=True)
    return grouped


def temporal_split(
    df: pd.DataFrame,
    time_col: str,
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by chronological order."""
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = df_sorted.iloc[:n_train].copy()
    val = df_sorted.iloc[n_train : n_train + n_val].copy()
    test = df_sorted.iloc[n_train + n_val :].copy()
    return train, val, test


def preprocess_kuairand(
    data_path: str | Path,
    sample_size: Optional[int] = 200_000,
    random_state: int = 42,
    temporal_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    interactions_file: str = "interactions.csv",
    users_file: str = "user_features.csv",
    items_file: str = "video_features.csv",
    user_col: str = "user_id",
    item_col: str = "video_id",
    label_col: str = "is_click",
    tag_col: str = "tag",
    categorical_cols: Optional[Sequence[str]] = None,
    numeric_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder], MultiLabelBinarizer, pd.DataFrame]:
    """
    Shared KuaiRand preprocessing pipeline.

    Returns:
        train, val, test:
            split DataFrames
        label_encoders:
            category encoders for selected categorical columns
        tag_mlb:
            MultiLabelBinarizer for item tags
        tag_matrix:
            item-level multi-hot DataFrame indexed by item id
    """
    tables = load_kuairand_tables(
        data_path=data_path,
        interactions_file=interactions_file,
        users_file=users_file,
        items_file=items_file,
    )

    interactions = tables.interactions.copy()
    if sample_size is not None and sample_size > 0 and sample_size < len(interactions):
        if label_col in interactions.columns:
            interactions = _stratified_downsample(interactions, label_col, sample_size, random_state)
        else:
            interactions = interactions.sample(sample_size, random_state=random_state).reset_index(drop=True)

    # Merge after sampling to avoid expensive full-table joins on large raw logs.
    df = (
        interactions.merge(tables.users, on=user_col, how="left")
        .merge(tables.items, on=item_col, how="left")
        .copy()
    )

    # Parse item tags and build item-level multi-hot matrix.
    if tag_col not in df.columns:
        alt_tag_col = _pick_existing_column(df, ["tags", "tag_ids", "video_tag", "video_tags"])
        if alt_tag_col is not None:
            tag_col = alt_tag_col
    if tag_col not in df.columns:
        raise KeyError(f"Expected tag column '{tag_col}' not found in data.")

    df["tag_list"] = df[tag_col].apply(parse_tag_ids)
    item_tags = (
        df[[item_col, "tag_list"]]
        .drop_duplicates(subset=[item_col])
        .sort_values(item_col)
        .reset_index(drop=True)
    )
    tag_mlb = MultiLabelBinarizer()
    tag_vectors = tag_mlb.fit_transform(item_tags["tag_list"])
    tag_matrix = pd.DataFrame(tag_vectors, index=item_tags[item_col], columns=tag_mlb.classes_)
    tag_matrix.index.name = item_col

    # Auto-detect columns if not explicitly provided.
    if categorical_cols is None:
        categorical_cols = [
            c
            for c in df.columns
            if c not in {user_col, item_col, label_col, tag_col, "tag_list"}
            and (df[c].dtype == "object" or str(df[c].dtype).startswith("category"))
        ]
    if numeric_cols is None:
        numeric_cols = []

    # Label encode selected categorical columns.
    label_encoders: Dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        series = df[col].astype(str).fillna("UNKNOWN")
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(series)
        label_encoders[col] = encoder

    # Scale selected numeric columns (if provided).
    existing_numeric_cols = [c for c in numeric_cols if c in df.columns]
    if existing_numeric_cols:
        scaler = StandardScaler()
        df[existing_numeric_cols] = scaler.fit_transform(df[existing_numeric_cols])

    # Find a usable time column for temporal split.
    time_col = _pick_existing_column(
        df, ["timestamp", "time_ms", "ts", "event_time", "date", "play_time_ms"]
    )
    if time_col is None:
        # Fall back to random split if no time information exists.
        shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        n = len(shuffled)
        n_train = int(n * temporal_ratios[0])
        n_val = int(n * temporal_ratios[1])
        train = shuffled.iloc[:n_train].copy()
        val = shuffled.iloc[n_train : n_train + n_val].copy()
        test = shuffled.iloc[n_train + n_val :].copy()
    else:
        train, val, test = temporal_split(df, time_col=time_col, ratios=temporal_ratios)

    return train, val, test, label_encoders, tag_mlb, tag_matrix
