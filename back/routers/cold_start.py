"""
routers/cold_start.py  —  cold-start module
=============================================

Four-phase progressive cold-start (fully deterministic, no random sampling):

  Phase 0  clicks = 0          Pure global popularity fallback
  Phase 1  1 ≤ clicks ≤ 7     Cluster confidence weighting: Pop×(1-conf) + Cluster×conf
  Phase 2  8 ≤ clicks ≤ 19    Cluster dominates: 0.6×Cluster + 0.4×KNN
  Phase 3  clicks ≥ 20        Deep fusion: 0.4×Cluster + 0.6×KNN neighbors

Design notes:
  - Phase 1: low early confidence → results near popularity, smoothly sliding toward Cluster as clicks grow
  - Phase 2: candidates come from the nearest K-Means cluster, diverging from global popularity
  - Phase 3: retains cluster assignment + KNN neighbor fusion for strong explainability
  - Same input always produces same output, convenient for demo reproducibility

POST /api/cold-start/recommend
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from utils.loader import (
    get_video_tags, tag_name,
    get_popular_candidates,
    get_cluster_candidates,
    knn_candidates_from_history,
    knn_score_from_history,
)

router = APIRouter()

CLUSTER_THRESHOLD = 8    # Phase 2 起始点击数
PHASE3_THRESHOLD  = 20   # Phase 3 起始点击数


class ColdStartRequest(BaseModel):
    clicked_videos: list[int] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)


def _phase0(top_k: int) -> list[dict]:
    """Phase 0: pure popularity, no personalization."""
    candidates = get_popular_candidates(n=top_k * 3)
    results = []
    for rank, vid in enumerate(candidates[:top_k]):
        pop_score = 1.0 - rank / (top_k * 3)
        results.append({
            "video_id": vid,
            "score": round(pop_score, 4),
            "pop_score": round(pop_score, 4),
            "knn_score": 0.0,
            "tag_boost": 0.0,
            "cluster_score": 0.0,
            "tags": [tag_name(t) for t in get_video_tags(vid)[:3]],
            "phase": 0,
        })
    return results


def _phase1(clicked_videos: list[int], n_clicks: int, top_k: int) -> tuple[list[dict], int, float]:
    """
    Phase 1: cluster confidence-weighted blend.
    score = (1 - eff_conf) × pop_score + eff_conf × cluster_score

    eff_conf = confidence × (n_clicks / CLUSTER_THRESHOLD)
    - Scales down cluster weight at low click counts to prevent over-trusting a single click
    - 1 click:  eff_conf = conf × 0.125 → mostly popularity
    - 7 clicks: eff_conf = conf × 0.875 → mostly cluster
    """
    seen = set(clicked_videos)
    cluster_cands, cluster_id, confidence = get_cluster_candidates(
        clicked_videos, n=300, exclude=seen
    )
    # Scale cluster weight by click progress to avoid jumping to cluster on 1 click
    confidence = confidence * (n_clicks / CLUSTER_THRESHOLD)
    pop_cands = get_popular_candidates(n=300, exclude=seen | set(cluster_cands))

    merged = list(dict.fromkeys(cluster_cands + pop_cands))

    cluster_rank = {vid: i for i, vid in enumerate(cluster_cands)}
    pop_rank_map = {vid: i for i, vid in enumerate(pop_cands)}
    max_cluster = len(cluster_cands) or 1
    max_pop     = len(pop_cands) or 1

    results = []
    for vid in merged:
        c_rank      = cluster_rank.get(vid, max_cluster)
        cluster_scr = 1.0 - c_rank / max_cluster

        p_rank    = pop_rank_map.get(vid, max_pop)
        pop_scr   = 1.0 - p_rank / max_pop

        score = (1 - confidence) * pop_scr + confidence * cluster_scr
        results.append({
            "video_id": vid,
            "score": round(score, 4),
            "pop_score": round(pop_scr, 4),
            "knn_score": 0.0,
            "tag_boost": 0.0,
            "cluster_score": round(cluster_scr, 4),
            "tags": [tag_name(t) for t in get_video_tags(vid)[:3]],
            "phase": 1,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k], cluster_id, confidence


def _phase2(clicked_videos: list[int], n_clicks: int, top_k: int) -> tuple[list[dict], int, float]:
    """
    Phase 2: cluster candidates with KNN weight linearly ramping 0 → 0.4.
    click 8  → knn_w=0.0, pure cluster (smooth handoff from Phase 1)
    click 19 → knn_w=0.4, Cluster 60% + KNN 40%
    """
    seen = set(clicked_videos)
    candidates, cluster_id, confidence = get_cluster_candidates(
        clicked_videos, n=250, exclude=seen
    )

    t     = (n_clicks - CLUSTER_THRESHOLD) / (PHASE3_THRESHOLD - CLUSTER_THRESHOLD)
    knn_w = 0.4 * t   # 0.0 → 0.4

    results = []
    max_rank = len(candidates) or 1
    for rank, vid in enumerate(candidates):
        cluster_score = 1.0 - rank / max_rank
        knn_s    = knn_score_from_history(clicked_videos, vid)
        knn_norm = float(1.0 / (1.0 + np.exp(-knn_s))) if knn_s > 0 else 0.0
        score = (1 - knn_w) * cluster_score + knn_w * knn_norm
        results.append({
            "video_id": vid,
            "score": round(score, 4),
            "pop_score": 0.0,
            "knn_score": round(knn_norm, 4),
            "tag_boost": 0.0,
            "cluster_score": round(cluster_score, 4),
            "tags": [tag_name(t) for t in get_video_tags(vid)[:3]],
            "phase": 2,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k], cluster_id, confidence


def _phase3(clicked_videos: list[int], n_clicks: int, top_k: int) -> tuple[list[dict], int, float]:
    """
    Phase 3: Cluster ∪ KNN neighbor candidates; KNN weight ramps 0.4 → 0.6.
    click 20 → knn_w=0.40 (smooth handoff from Phase 2)
    click 30 → knn_w=0.60 (cap)
    """
    seen = set(clicked_videos)

    cluster_cands, cluster_id, confidence = get_cluster_candidates(
        clicked_videos, n=200, exclude=seen
    )
    knn_cands = knn_candidates_from_history(clicked_videos, n=200, exclude=seen)
    merged = list(dict.fromkeys(cluster_cands + knn_cands))
    if len(merged) < top_k:
        extra = get_popular_candidates(n=top_k * 2, exclude=seen | set(merged))
        merged = merged + extra

    t3    = min(1.0, (n_clicks - PHASE3_THRESHOLD) / 10)
    knn_w = 0.4 + 0.2 * t3   # 0.4 → 0.6

    cluster_rank = {vid: i for i, vid in enumerate(cluster_cands)}
    max_cluster  = len(cluster_cands) or 1

    results = []
    for vid in merged:
        c_rank        = cluster_rank.get(vid, max_cluster)
        cluster_score = 1.0 - c_rank / max_cluster
        knn_s    = knn_score_from_history(clicked_videos, vid)
        knn_norm = float(1.0 / (1.0 + np.exp(-knn_s))) if knn_s > 0 else 0.0
        score = (1 - knn_w) * cluster_score + knn_w * knn_norm
        results.append({
            "video_id": vid,
            "score": round(score, 4),
            "pop_score": 0.0,
            "knn_score": round(knn_norm, 4),
            "tag_boost": 0.0,
            "cluster_score": round(cluster_score, 4),
            "tags": [tag_name(t) for t in get_video_tags(vid)[:3]],
            "phase": 3,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k], cluster_id, confidence


@router.post("/cold-start/recommend")
def cold_start_recommend(req: ColdStartRequest):
    n_clicks   = len(req.clicked_videos)
    cluster_id = -1
    confidence = 0.0

    if n_clicks == 0:
        results    = _phase0(req.top_k)
        phase_label = "Phase 0 · Popularity Fallback"

    elif n_clicks < CLUSTER_THRESHOLD:
        results, cluster_id, confidence = _phase1(req.clicked_videos, n_clicks, req.top_k)
        phase_label = f"Phase 1 · Cluster #{cluster_id} (conf {confidence:.0%})"

    elif n_clicks < PHASE3_THRESHOLD:
        results, cluster_id, confidence = _phase2(req.clicked_videos, n_clicks, req.top_k)
        phase_label = f"Phase 2 · Cluster #{cluster_id} (conf {confidence:.0%})"

    else:
        results, cluster_id, confidence = _phase3(req.clicked_videos, n_clicks, req.top_k)
        phase_label = f"Phase 3 · Cluster #{cluster_id} + KNN Fusion"

    progress_pct = round(min(n_clicks / PHASE3_THRESHOLD, 1.0), 4)

    return {
        "n_clicks":     n_clicks,
        "cluster_id":   cluster_id,
        "confidence":   round(confidence, 4),
        "progress_pct": progress_pct,
        "phase_label":  phase_label,
        "results":      results,
    }
