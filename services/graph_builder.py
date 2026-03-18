from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_shared = (Path(__file__).resolve().parent.parent / "../recsys-shared").resolve()
if str(_shared) not in sys.path:
    sys.path.insert(0, str(_shared))

from interest_graph.graph_data import InterestLink, InterestNode
from tag_display import get_tag_display_name


def build_interest_graph(
    user_id: int,
    interactions_df: pd.DataFrame,
    tag_matrix: pd.DataFrame,
    lambda_decay: float = 0.035,
    mutation_z_threshold: float = 2.0,
    top_k_tags: int = 10,
    top_k_edges: int = 12,
) -> tuple[list[InterestNode], list[InterestLink]]:
    """Build interest-evolution graph data for a single user.

    Parameters
    ----------
    user_id : int
        Target user whose interest graph will be constructed.
    interactions_df : pd.DataFrame
        Full interaction log containing at least ``user_id``, ``video_id``,
        and a time column (``time_ms`` preferred).
    tag_matrix : pd.DataFrame
        Multi-hot matrix indexed by ``video_id``; columns are integer tag IDs,
        values are 0/1.
    lambda_decay : float
        Exponential decay rate per day applied to recency weighting.
    mutation_z_threshold : float
        Z-score above which a tag is flagged as a sudden-spike *mutation*.
    top_k_tags : int
        Maximum tag nodes returned.
    top_k_edges : int
        Maximum transition links returned.
    """
    # ── 1. filter to user & sort by time ─────────────────────────────────
    udf = interactions_df[interactions_df["user_id"] == user_id].copy()
    if udf.empty:
        return [], []

    time_col = next(
        (c for c in ("time_ms", "timestamp", "ts", "date") if c in udf.columns),
        None,
    )
    if time_col is None:
        udf["_seq"] = range(len(udf))
        time_col = "_seq"
    udf = udf.sort_values(time_col).reset_index(drop=True)

    # ── 2. map video_id → tags via tag_matrix ────────────────────────────
    valid_vids = tag_matrix.index.intersection(udf["video_id"].unique())
    if valid_vids.empty:
        return [], []

    sub = tag_matrix.loc[valid_vids].copy()
    sub.index.name = "video_id"

    tag_long = (
        sub.reset_index()
        .melt(id_vars="video_id", var_name="tag_id", value_name="_v")
        .query("_v == 1")
        .drop(columns="_v")
    )
    tag_df = udf.merge(tag_long, on="video_id", how="inner")
    if tag_df.empty:
        return [], []

    tag_df["tag_id"] = tag_df["tag_id"].astype(int)
    min_t = float(tag_df[time_col].min())
    max_t = float(tag_df[time_col].max())
    t_range = max_t - min_t if max_t > min_t else 1.0

    # ── 3. per-tag aggregation ───────────────────────────────────────────
    aggs: dict[str, tuple[str, str]] = {
        "count": (time_col, "size"),
        "last_time": (time_col, "max"),
        "first_time": (time_col, "min"),
    }
    eng_cols_present: list[str] = []
    for c in ("is_click", "is_like", "is_follow", "is_comment", "is_forward"):
        if c in tag_df.columns:
            aggs[c] = (c, "sum")
            eng_cols_present.append(c)
    if "play_time_ms" in tag_df.columns:
        aggs["avg_play_ms"] = ("play_time_ms", "mean")

    tag_stats = tag_df.groupby("tag_id").agg(**aggs).reset_index()

    # ── 4. decay = exp(-λ * Δdays) ──────────────────────────────────────
    tag_stats["delta_days"] = (max_t - tag_stats["last_time"]) / 86_400_000.0
    tag_stats["decay"] = np.exp(-lambda_decay * tag_stats["delta_days"]).clip(0.0, 1.0)

    # ── 5. timestamp (normalised first-appearance → [0.05, 0.95]) ───────
    tag_stats["timestamp"] = 0.05 + 0.9 * (tag_stats["first_time"] - min_t) / t_range

    # ── 6. mutation detection (short 3-day vs long 90-day window) ───────
    short_cut = max_t - 3 * 86_400_000
    long_cut = max_t - 90 * 86_400_000
    s_cnt = tag_df.loc[tag_df[time_col] >= short_cut].groupby("tag_id").size()
    l_cnt = tag_df.loc[tag_df[time_col] >= long_cut].groupby("tag_id").size()
    tot_s = max(1, int(s_cnt.sum()))
    tot_l = max(1, int(l_cnt.sum()))

    mutation_ids: set[int] = set()
    for tid in s_cnt.index:
        short_rate = s_cnt.get(tid, 0) / tot_s
        long_count = l_cnt.get(tid, 0)
        long_rate = long_count / tot_l
        denom = max(0.01, math.sqrt(long_rate * (1 - long_rate) / max(1, long_count)))
        z = (short_rate - long_rate) / denom
        if z > mutation_z_threshold and long_count < 5:
            mutation_ids.add(int(tid))

    # ── 7. status classification ────────────────────────────────────────
    conditions = [
        tag_stats["tag_id"].isin(mutation_ids),
        tag_stats["decay"] > 0.7,
    ]
    choices = ["mutation", "active"]
    tag_stats["status"] = np.select(conditions, choices, default="fading")

    top = tag_stats.nlargest(top_k_tags, "count").copy()
    top_set: set[int] = set(top["tag_id"])
    active_set: set[int] = set(top.loc[top["status"] == "active", "tag_id"])
    user_all_tags: set[int] = set(tag_stats["tag_id"])

    # predicted nodes: co-occur with active tags globally but unseen by user
    if active_set:
        cols_active = [c for c in active_set if c in tag_matrix.columns]
        if cols_active:
            active_mask = tag_matrix[cols_active].max(axis=1) == 1
            neighbor_sums = tag_matrix.loc[active_mask].sum(axis=0)
            neighbor_tags = {int(c) for c in tag_matrix.columns[neighbor_sums > 0]}
            predicted = neighbor_tags - user_all_tags
            slots = max(2, top_k_tags - len(top))
            for ptid in list(predicted)[:slots]:
                pred_row: dict = {
                    "tag_id": ptid,
                    "count": 0,
                    "last_time": max_t,
                    "first_time": max_t,
                    "delta_days": 0.0,
                    "decay": 0.65,
                    "timestamp": 0.92,
                    "status": "predicted",
                }
                for c in eng_cols_present:
                    pred_row[c] = 0
                if "avg_play_ms" in top.columns:
                    pred_row["avg_play_ms"] = 0.0
                top = pd.concat([top, pd.DataFrame([pred_row])], ignore_index=True)
            top_set = set(top["tag_id"])

    # ── 8. transition probabilities from sequential tag pairs ───────────
    vid_to_tags: dict[int, list[int]] = {}
    for vid in udf["video_id"].unique():
        if vid not in tag_matrix.index:
            continue
        row = tag_matrix.loc[vid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        tags = [int(t) for t in row[row == 1].index if int(t) in top_set]
        if tags:
            vid_to_tags[int(vid)] = tags

    trans: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    prev_tags: list[int] | None = None
    for vid in udf["video_id"]:
        cur = vid_to_tags.get(int(vid))
        if cur is not None and prev_tags is not None:
            for s in prev_tags:
                for t in cur:
                    if s != t:
                        trans[s][t] += 1
        if cur is not None:
            prev_tags = cur

    ts_map = dict(zip(top["tag_id"].astype(int), top["timestamp"].astype(float)))
    links: list[InterestLink] = []
    for src, tgts in trans.items():
        total = sum(tgts.values())
        for tgt, cnt in tgts.items():
            prob = cnt / total
            avg_ts = (ts_map.get(src, 0.5) + ts_map.get(tgt, 0.5)) / 2.0
            links.append(InterestLink(str(src), str(tgt), prob, timestamp=avg_ts))

    links.sort(key=lambda lk: lk.probability, reverse=True)
    links = links[:top_k_edges]

    # ── 9-10. build InterestNode list with metrics ──────────────────────
    _eng_metric_cols = [
        c for c in ("is_like", "is_follow", "is_comment", "is_forward") if c in top.columns
    ]

    nodes: list[InterestNode] = []
    for _, r in top.iterrows():
        tid = int(r["tag_id"])
        metrics = None
        if r["status"] == "active" and r["count"] > 0:
            eng = sum(float(r[c]) for c in _eng_metric_cols) / r["count"] * 100
            ctr = float(r.get("is_click", 0)) / r["count"] * 100
            watch = float(r.get("avg_play_ms", 0)) / 1000.0
            metrics = {
                "ENG": f"{eng:.0f}%",
                "CTR": f"{ctr:.0f}%",
                "WATCH": f"{watch:.0f}s",
            }
        nodes.append(
            InterestNode(
                id=str(tid),
                label=get_tag_display_name(tid),
                decay=float(r["decay"]),
                status=r["status"],
                tags=[tid],
                timestamp=float(r["timestamp"]),
                metrics=metrics,
            )
        )

    return nodes, links


# ─────────────────────────────────────────────────────────────────────────
# Convenience loader
# ─────────────────────────────────────────────────────────────────────────


def build_demo_from_kuairand(
    data_dir: str | Path,
    user_id: int | None = None,
    sample_rows: int = 250_000,
) -> tuple[list[InterestNode], list[InterestLink]]:
    """Load KuaiRand-1K data and build an interest graph.

    If *user_id* is ``None``, the most-active user is chosen automatically.
    """
    from data_pipeline import load_kuairand_tables, parse_tag_ids
    from sklearn.preprocessing import MultiLabelBinarizer
    from tag_display import ensure_tag_mapping

    tables = load_kuairand_tables(data_dir)
    ensure_tag_mapping(data_dir)
    interactions = tables.interactions.copy()

    if sample_rows and sample_rows < len(interactions):
        interactions = interactions.head(sample_rows)

    if user_id is None:
        user_id = int(interactions["user_id"].value_counts().idxmax())

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

    return build_interest_graph(
        user_id=user_id,
        interactions_df=interactions,
        tag_matrix=tag_matrix,
    )
