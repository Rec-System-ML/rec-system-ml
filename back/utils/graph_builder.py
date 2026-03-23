"""
utils/graph_builder.py
----------------------
Module 2: Interest Evolution Graph — 10-step core algorithm.

Ported from rec-system-ml-feature-interest_evo/services/graph_builder.py.
Adapted for the main dashboard backend (no shared_bootstrap dependency).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class InterestNode:
    id: str
    label: str
    decay: float                        # [0,1] freshness (controls D3 node size)
    status: str                         # "active" | "fading" | "predicted" | "mutation"
    tags: list = field(default_factory=list)
    timestamp: Optional[float] = None  # [0,1] timeline position
    metrics: Optional[dict] = None     # ENG/CTR/WATCH for active nodes

    def to_dict(self) -> dict:
        return {
            "id":        self.id,
            "label":     self.label,
            "decay":     round(self.decay, 4),
            "status":    self.status,
            "tags":      self.tags or [],
            "timestamp": round(self.timestamp, 4) if self.timestamp is not None else None,
            "metrics":   self.metrics,
        }


@dataclass
class InterestLink:
    source: str
    target: str
    probability: float                  # [0,1] 1st-order Markov transition probability
    timestamp: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "source":      self.source,
            "target":      self.target,
            "probability": round(self.probability, 4),
            "timestamp":   round(self.timestamp, 4) if self.timestamp is not None else None,
        }


# ── Main algorithm ─────────────────────────────────────────────────────────────

def build_interest_graph(
    user_id: int,
    interactions_df: pd.DataFrame,
    tag_matrix: pd.DataFrame,
    tag_name_fn: Optional[Callable[[int], str]] = None,
    lambda_decay: float = 0.035,
    mutation_z_threshold: float = 2.0,
    top_k_tags: int = 10,
    top_k_edges: int = 12,
) -> tuple[list[InterestNode], list[InterestLink]]:
    """
    Build an interest-evolution graph for a single user (10-step algorithm).

    Parameters
    ----------
    user_id        : target user
    interactions_df: full interaction log (must have user_id, video_id, + time col)
    tag_matrix     : multi-hot DataFrame indexed by video_id; columns = tag IDs (int)
    tag_name_fn    : tag_id -> display name; defaults to "Tag-{id}"
    lambda_decay   : exponential decay rate (1/day); default 0.035 → ~20-day half-life
    mutation_z_threshold : Z-score threshold for mutation detection (default 2.0)
    top_k_tags     : max nodes in the graph (default 10)
    top_k_edges    : max edges in the graph (default 12)
    """
    if tag_name_fn is None:
        tag_name_fn = lambda tid: f"Tag-{tid}"

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 01]  Filter target user, sort chronologically
    # ══════════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 02]  Expand video → tag mapping (multi-hot → long format)
    # ══════════════════════════════════════════════════════════════════════════
    valid_vids = tag_matrix.index.intersection(udf["video_id"].unique())
    if len(valid_vids) == 0:
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

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 03]  Aggregate tag statistics
    # ══════════════════════════════════════════════════════════════════════════
    aggs: dict = {
        "count":      (time_col, "size"),
        "last_time":  (time_col, "max"),
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

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 04]  Exponential time decay:  decay = exp(−λ × Δdays)
    #
    #   λ=0.035 → half-life ≈ 20 days
    #   decay ∈ [0,1]: 1 = very fresh, 0 = stale
    #   Used by D3.js to control node size
    # ══════════════════════════════════════════════════════════════════════════
    tag_stats["delta_days"] = (max_t - tag_stats["last_time"]) / 86_400_000.0
    tag_stats["decay"] = np.exp(-lambda_decay * tag_stats["delta_days"]).clip(0.0, 1.0)

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 05]  Normalize first-appearance timestamp → [0.05, 0.95]
    #
    #   Controls the horizontal position on the D3 timeline:
    #   0.05 = very old interest, 0.95 = very recent interest
    # ══════════════════════════════════════════════════════════════════════════
    tag_stats["timestamp"] = 0.05 + 0.9 * (tag_stats["first_time"] - min_t) / t_range

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 06]  Mutation detection (binomial proportion Z-test)
    #
    #   Compare short-window (3-day) ratio vs long-window (90-day) baseline.
    #   Formula: Z = (r_short − r_long) / σ_long
    #   Flag as mutation if: Z > threshold AND 90-day count < 5
    # ══════════════════════════════════════════════════════════════════════════
    short_cut = max_t - 3  * 86_400_000
    long_cut  = max_t - 90 * 86_400_000

    s_cnt = tag_df.loc[tag_df[time_col] >= short_cut].groupby("tag_id").size()
    l_cnt = tag_df.loc[tag_df[time_col] >= long_cut ].groupby("tag_id").size()
    tot_s = max(1, int(s_cnt.sum()))
    tot_l = max(1, int(l_cnt.sum()))

    mutation_ids: set[int] = set()
    for tid in s_cnt.index:
        short_rate = s_cnt.get(tid, 0) / tot_s
        long_count = l_cnt.get(tid, 0)
        long_rate  = long_count / tot_l
        denom = max(0.01, math.sqrt(long_rate * (1 - long_rate) / max(1, long_count)))
        z = (short_rate - long_rate) / denom
        if z > mutation_z_threshold and long_count < 5:
            mutation_ids.add(int(tid))

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 07]  Node status classification (priority: mutation > active > fading)
    #
    #   active  (green) : last_time ≥ 60th percentile (relatively recent)
    #   fading  (gray)  : older interests
    #   mutation(red)   : sudden burst detected by Z-test
    # ══════════════════════════════════════════════════════════════════════════
    tag_stats["status"] = "fading"
    tag_stats["_recency_pct"] = tag_stats["last_time"].rank(method="average", pct=True)
    tag_stats.loc[tag_stats["_recency_pct"] >= 0.60, "status"] = "active"
    tag_stats.loc[tag_stats["tag_id"].isin(mutation_ids), "status"] = "mutation"

    # Guardrail: if all non-mutation nodes are active, force oldest 30% to fading
    non_mut_mask = ~tag_stats["tag_id"].isin(mutation_ids)
    if non_mut_mask.sum() >= 3 and tag_stats.loc[non_mut_mask, "status"].eq("active").all():
        oldest_n = max(1, int(non_mut_mask.sum() * 0.3))
        oldest_idx = tag_stats.loc[non_mut_mask].nsmallest(oldest_n, "last_time").index
        tag_stats.loc[oldest_idx, "status"] = "fading"

    top = tag_stats.nlargest(top_k_tags, "count").copy()

    # Ensure at least one fading node for visual contrast
    if not top["status"].eq("fading").any():
        fading_pool = tag_stats.loc[tag_stats["status"] == "fading"].sort_values(
            "count", ascending=False
        )
        if not fading_pool.empty:
            replace_idx = top.loc[top["status"] != "mutation", "count"].idxmin()
            top.loc[replace_idx, top.columns] = fading_pool.iloc[0][top.columns].values

    top_set: set[int]    = set(top["tag_id"])
    active_set: set[int] = set(top.loc[top["status"] == "active", "tag_id"])
    user_all_tags: set[int] = set(tag_stats["tag_id"])

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 08]  Predicted nodes (purple) — co-occurrence based
    #
    #   Find videos that share active tags, collect co-occurring tags the user
    #   hasn't seen (or rarely seen) → mark as "predicted" interest.
    #   Fixed: decay=0.65, timestamp=0.85 (right side of timeline = future)
    # ══════════════════════════════════════════════════════════════════════════
    if active_set:
        cols_active = [c for c in active_set if c in tag_matrix.columns]
        if cols_active:
            active_mask   = tag_matrix[cols_active].max(axis=1) == 1
            neighbor_sums = tag_matrix.loc[active_mask].sum(axis=0)
            neighbor_sums = neighbor_sums[neighbor_sums > 0].sort_values(ascending=False)

            predicted_candidates = [
                int(c) for c in neighbor_sums.index if int(c) not in user_all_tags
            ]
            if not predicted_candidates:
                predicted_candidates = [
                    int(c) for c in neighbor_sums.index if int(c) not in active_set
                ]
            if not predicted_candidates:
                global_pop = tag_matrix.sum(axis=0).sort_values(ascending=False)
                predicted_candidates = [
                    int(c) for c in global_pop.index if int(c) not in active_set
                ]

            slots = max(2, top_k_tags - len(top))
            for ptid in predicted_candidates[:slots]:
                pred_row: dict = {
                    "tag_id": ptid, "count": 0,
                    "last_time": max_t, "first_time": max_t,
                    "delta_days": 0.0, "decay": 0.65,
                    "timestamp": 0.85, "status": "predicted",
                }
                for c in eng_cols_present:
                    pred_row[c] = 0
                if "avg_play_ms" in top.columns:
                    pred_row["avg_play_ms"] = 0.0
                top = pd.concat([top, pd.DataFrame([pred_row])], ignore_index=True)
            top_set = set(top["tag_id"])

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 09]  First-order Markov transition probabilities → InterestLink
    #
    #   Scan watch sequence; count adjacent tag transitions.
    #   P(A→B) = count(A→B) / Σ count(A→*)
    #   Keep top_k_edges by probability.
    # ══════════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════════
    # [Step 10]  Assemble InterestNode list with metrics
    #
    #   Active nodes get ENG / CTR / WATCH metrics:
    #     ENG   = (like+follow+comment+forward) / count × 100%
    #     CTR   = is_click / count × 100%
    #     WATCH = avg_play_ms / 1000 (seconds)
    # ══════════════════════════════════════════════════════════════════════════
    _eng_metric_cols = [
        c for c in ("is_like", "is_follow", "is_comment", "is_forward")
        if c in top.columns
    ]

    nodes: list[InterestNode] = []
    for _, r in top.iterrows():
        tid = int(r["tag_id"])
        metrics = None
        if r["status"] == "active" and r["count"] > 0:
            eng   = sum(float(r[c]) for c in _eng_metric_cols) / r["count"] * 100
            ctr   = float(r.get("is_click", 0)) / r["count"] * 100
            watch = float(r.get("avg_play_ms", 0)) / 1000.0
            metrics = {
                "ENG":   f"{eng:.0f}%",
                "CTR":   f"{ctr:.0f}%",
                "WATCH": f"{watch:.0f}s",
            }
        nodes.append(
            InterestNode(
                id=str(tid),
                label=tag_name_fn(tid),
                decay=float(r["decay"]),
                status=r["status"],
                tags=[tid],
                timestamp=float(r["timestamp"]),
                metrics=metrics,
            )
        )

    return nodes, links
