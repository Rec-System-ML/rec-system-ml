"""
routers/recommend.py
--------------------
GET  /api/popular              → global popular Top-N
POST /api/recommend            → returning-user recommendation (ItemKNN + XGBoost blend)
POST /api/recommend/realtime   → returning-user real-time re-ranking (with session clicks)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import utils.loader as _loader
from utils.loader import get_artifact, get_video_tags, tag_name

router = APIRouter()


class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = Field(default=10, ge=1, le=50)
    alpha: float = Field(default=0.6, ge=0.0, le=1.0,
                         description="XGBoost 权重，1-alpha 为 KNN 权重")


class VideoResult(BaseModel):
    rank: int
    video_id: int
    score: float
    knn_score: float
    ctr_score: float
    tags: list[str]
    explain: str


def _build_candidate_frame(user_id: int, candidates: list[int]) -> pd.DataFrame:
    now_ms = int(pd.Timestamp.now().value // 1_000_000)
    return pd.DataFrame({
        "user_id": [user_id] * len(candidates),
        "video_id": candidates,
        "time_ms": [now_ms] * len(candidates),
    })


def _run_recommend(user_id: int, top_k: int, alpha: float) -> list[VideoResult]:
    art = get_artifact()
    item_knn    = art["item_knn"]
    ctr_model   = art["ctr_model"]
    feat_builder = art["feature_builder"]
    reranker    = art["reranker"]

    candidates = item_knn.candidate_pool(user_id=user_id, top_pop_n=800)
    if not candidates:
        return []

    cand_df = _build_candidate_frame(user_id, candidates)
    cand_df["knn_score"] = cand_df.apply(
        lambda r: item_knn.score(int(r["user_id"]), int(r["video_id"])), axis=1
    )
    X = feat_builder.transform(cand_df)
    cand_df["ctr_score"]  = ctr_model.predict_proba(X)
    knn_prob = 1.0 / (1.0 + np.exp(-cand_df["knn_score"]))
    cand_df["prediction"] = alpha * cand_df["ctr_score"] + (1 - alpha) * knn_prob

    reranked = reranker.rerank_dataframe(
        cand_df, score_col="prediction", time_col="time_ms", top_k=top_k
    )

    results = []
    for rank, (_, row) in enumerate(reranked.iterrows(), start=1):
        vid = int(row["video_id"])
        results.append(VideoResult(
            rank=rank,
            video_id=vid,
            score=round(float(row.get("rerank_score", row["prediction"])), 4),
            knn_score=round(float(row["knn_score"]), 4),
            ctr_score=round(float(row["ctr_score"]), 4),
            tags=[tag_name(t) for t in get_video_tags(vid)[:5]],
            explain=item_knn.explain(user_id, vid),
        ))
    return results


@router.get("/popular")
def get_popular(n: int = 10):
    """Global popular Top-N, sorted by click count in the training log."""
    results = []
    for rank, (vid, cnt) in enumerate(_loader.popularity_ranking[:n], start=1):
        results.append({
            "rank": rank,
            "video_id": vid,
            "click_count": cnt,
            "tags": [tag_name(t) for t in get_video_tags(vid)[:3]],
        })
    return {"results": results}


@router.post("/recommend")
def recommend(req: RecommendRequest):
    art = get_artifact()
    if req.user_id not in art.get("train_users", []):
        raise HTTPException(status_code=404, detail=f"user_id {req.user_id} not found")
    results = _run_recommend(req.user_id, req.top_k, req.alpha)
    return {"user_id": req.user_id, "top_k": req.top_k, "alpha": req.alpha,
            "results": [r.model_dump() for r in results]}


class RealtimeRequest(BaseModel):
    user_id: int
    extra_clicks: list[int] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)


@router.post("/recommend/realtime")
def recommend_realtime(req: RealtimeRequest):
    """Returning-user real-time recommendation: re-score with session clicks on top of training history."""
    art = get_artifact()
    if req.user_id not in art.get("train_users", []):
        raise HTTPException(status_code=404, detail=f"user_id {req.user_id} not found")

    item_knn     = art["item_knn"]
    ctr_model    = art["ctr_model"]
    feat_builder = art["feature_builder"]
    reranker     = art["reranker"]

    # Candidates: training history pool + KNN neighbors of session clicks
    base_candidates = set(item_knn.candidate_pool(user_id=req.user_id, top_pop_n=800))
    for vid in req.extra_clicks:
        for neigh, _ in item_knn.item_neighbor_scores.get(vid, [])[:50]:
            base_candidates.add(neigh)
    candidates = [v for v in base_candidates if v not in set(req.extra_clicks)]
    if not candidates:
        return {"user_id": req.user_id, "extra_clicks": len(req.extra_clicks), "results": []}

    # Session click weight grows linearly with click count, capped at 0.5
    extra_weight = min(len(req.extra_clicks) / 10, 0.5)

    trained_scores: dict[int, float] = {}
    for hist_vid in item_knn.user_histories.get(req.user_id, set()):
        for neigh_vid, sim in item_knn.item_neighbor_scores.get(hist_vid, []):
            trained_scores[neigh_vid] = trained_scores.get(neigh_vid, 0.0) + sim

    extra_scores: dict[int, float] = {}
    for hist_vid in req.extra_clicks:
        for neigh_vid, sim in item_knn.item_neighbor_scores.get(hist_vid, []):
            extra_scores[neigh_vid] = extra_scores.get(neigh_vid, 0.0) + sim

    cand_df = _build_candidate_frame(req.user_id, candidates)
    cand_df["knn_score"] = cand_df["video_id"].map(
        lambda vid: trained_scores.get(vid, 0.0) + extra_weight * extra_scores.get(vid, 0.0)
    )
    X = feat_builder.transform(cand_df)
    cand_df["ctr_score"]  = ctr_model.predict_proba(X)
    knn_prob = 1.0 / (1.0 + np.exp(-cand_df["knn_score"]))
    cand_df["prediction"] = req.alpha * cand_df["ctr_score"] + (1 - req.alpha) * knn_prob

    reranked = reranker.rerank_dataframe(
        cand_df, score_col="prediction", time_col="time_ms", top_k=req.top_k
    )

    results = []
    for rank, (_, row) in enumerate(reranked.iterrows(), start=1):
        vid = int(row["video_id"])
        results.append({
            "rank": rank,
            "video_id": vid,
            "score": round(float(row.get("rerank_score", row["prediction"])), 4),
            "knn_score": round(float(row["knn_score"]), 4),
            "ctr_score": round(float(row["ctr_score"]), 4),
            "tags": [tag_name(t) for t in get_video_tags(vid)[:3]],
        })
    return {"user_id": req.user_id, "extra_clicks": len(req.extra_clicks), "results": results}
