"""Tag-based video candidate expansion for Module 2 → Module 1 integration."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def videos_by_tags(
    tag_matrix: pd.DataFrame,
    tag_ids: List[int],
    min_tags: int = 1,
    max_videos: int = 200,
) -> List[int]:
    """Return video IDs that have at least min_tags of the given tags.

    Videos matching more tags are ranked higher (more relevant).
    """
    if not tag_ids:
        return []
    valid_cols = [t for t in tag_ids if t in tag_matrix.columns]
    if not valid_cols:
        return []
    hits = tag_matrix[valid_cols].sum(axis=1)
    candidates = hits[hits >= min_tags].sort_values(ascending=False)
    return candidates.head(max_videos).index.tolist()


def _load_tag_matrix_and_interactions(
    data_dir: str | Path,
    sample_rows: int = 250_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load interactions and tag_matrix from KuaiRand data dir."""
    import sys
    _shared = (Path(__file__).resolve().parent.parent / "../recsys-shared").resolve()
    if _shared.exists() and str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

    from data_pipeline import load_kuairand_tables, parse_tag_ids
    from sklearn.preprocessing import MultiLabelBinarizer
    from tag_display import ensure_tag_mapping

    tables = load_kuairand_tables(data_dir)
    ensure_tag_mapping(data_dir)
    interactions = tables.interactions.copy()

    if sample_rows and sample_rows < len(interactions):
        interactions = interactions.head(sample_rows)

    items = tables.items.copy()
    tag_col = next(
        (c for c in ("tag", "tags", "tag_ids", "video_tag") if c in items.columns),
        None,
    )
    if tag_col is None:
        raise KeyError("No tag column found in items table")

    items["_tags"] = items[tag_col].apply(parse_tag_ids)
    items = items.drop_duplicates(subset=["video_id"])

    mlb = MultiLabelBinarizer()
    vectors = mlb.fit_transform(items["_tags"])
    tag_matrix = pd.DataFrame(vectors, index=items["video_id"], columns=mlb.classes_)
    tag_matrix.index.name = "video_id"

    return interactions, tag_matrix


def predicted_tag_videos(
    user_id: int,
    data_dir: str | Path,
    sample_rows: int = 250_000,
    max_videos_per_tag: int = 100,
) -> List[int]:
    """Get video IDs for Module 2 predicted tags (interest expansion).

    Calls build_interest_graph, extracts predicted nodes, maps to videos via tag_matrix.
    Returns empty list if no predicted tags or data load fails.
    """
    from .graph_builder import build_interest_graph

    try:
        interactions, tag_matrix = _load_tag_matrix_and_interactions(
            data_dir=data_dir,
            sample_rows=sample_rows,
        )
    except Exception:
        return []

    nodes, _ = build_interest_graph(
        user_id=user_id,
        interactions_df=interactions,
        tag_matrix=tag_matrix,
    )

    predicted_tags = [int(n.id) for n in nodes if n.status == "predicted"]
    if not predicted_tags:
        return []

    return videos_by_tags(
        tag_matrix=tag_matrix,
        tag_ids=predicted_tags,
        min_tags=1,
        max_videos=max_videos_per_tag,
    )
