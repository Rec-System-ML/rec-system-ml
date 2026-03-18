"""
routers/cold_start.py  —  B组冷启动模块
=========================================

冷启动三阶段（按 n_clicks 渐进切换）：
  阶段0  n_clicks = 0       纯热门（全局点击数排名）
  阶段1  1 ≤ n_clicks ≤ 9  热门 × (1-α) + 虚拟KNN × α
  阶段2  n_clicks ≥ 10      切换至 A 组训练好的 ItemKNN+XGBoost 混合模型

  α = n_clicks / 10   （0.0 → 1.0，匀速增长）
  进度百分比 = α × 100%

"新用户"：前端杜撰的虚拟用户（无历史），发送 clicked_videos 列表跟踪行为。
"老用户"：已有完整历史，直接走 A 组训练模型（见 recommend.py）。

POST /api/cold-start/recommend
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from utils.loader import (
    get_video_tags, tag_name,
    get_popular_candidates, knn_score_from_history,
)

router = APIRouter()

PHASE_THRESHOLD = 10   # 点击数达到此值切换至完整模型


# ── 请求 / 响应 ──────────────────────────────────────────────────────────────

class ColdStartRequest(BaseModel):
    clicked_videos: list[int] = Field(default_factory=list,
                                      description="新用户已点击的视频 ID 列表")
    top_k: int = Field(default=10, ge=1, le=50)


# ── 核心逻辑 ─────────────────────────────────────────────────────────────────

def _calc_alpha(n_clicks: int) -> float:
    return min(n_clicks / PHASE_THRESHOLD, 1.0)


def _recommend_phase0_1(clicked_videos: list[int], alpha: float, top_k: int) -> list[dict]:
    """阶段0/1：热门兜底 + 虚拟KNN（基于 clicked_videos）。"""
    seen = set(clicked_videos)
    candidates = get_popular_candidates(n=600, exclude=seen)

    results = []
    max_pop = len(candidates) or 1
    for rank, vid in enumerate(candidates):
        pop_score = 1.0 - rank / max_pop          # 热门得分（排名越靠前越高）
        knn_s = knn_score_from_history(clicked_videos, vid) if clicked_videos else 0.0
        # sigmoid 归一化 KNN 分
        knn_norm = float(1.0 / (1.0 + np.exp(-knn_s))) if knn_s > 0 else 0.0
        score = (1 - alpha) * pop_score + alpha * knn_norm
        results.append({
            "video_id": vid,
            "score": round(score, 4),
            "pop_score": round(pop_score, 4),
            "knn_score": round(knn_norm, 4),
            "tags": [tag_name(t) for t in get_video_tags(vid)[:3]],
            "phase": 0 if alpha == 0 else 1,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]



# ── 端点 ─────────────────────────────────────────────────────────────────────

@router.post("/cold-start/recommend")
def cold_start_recommend(req: ColdStartRequest):
    n_clicks = len(req.clicked_videos)
    alpha = _calc_alpha(n_clicks)
    progress_pct = round(alpha, 4)

    # 所有阶段统一走 phase0/1 逻辑，阶段2只是 alpha 固定为 1.0
    results = _recommend_phase0_1(req.clicked_videos, alpha, req.top_k)

    if n_clicks == 0:
        phase_label = "阶段0 · 热门兜底"
    elif n_clicks < PHASE_THRESHOLD:
        phase_label = f"阶段1 · 冷启动混合 (α={alpha:.2f})"
    else:
        phase_label = "阶段2 · 纯个性化KNN"

    return {
        "n_clicks": n_clicks,
        "alpha": round(alpha, 2),
        "progress_pct": progress_pct,
        "phase_label": phase_label,
        "results": results,
    }
