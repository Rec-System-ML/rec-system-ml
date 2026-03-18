"""
routers/stats.py
----------------
GET /api/stats  →  Dashboard 首页 KPI 数据
"""
from fastapi import APIRouter
from utils.loader import get_artifact

router = APIRouter()


@router.get("/stats")
def get_stats():
    art = get_artifact()
    meta = art.get("data_meta", {})
    metrics = art.get("metrics", {})

    return {
        "user_count": 1000,
        "video_count": 4370000,
        "interaction_count": meta.get("rows_used", 0),
        "train_rows": meta.get("train_rows", 0),
        "auc": round(metrics.get("blend_test", {}).get("auc", 0), 4),
        "ndcg_at_10": round(metrics.get("ranking_test", {}).get("ndcg@10", 0), 4),
        "precision_at_10": round(metrics.get("ranking_test", {}).get("precision@10", 0), 4),
        "recall_at_10": round(metrics.get("ranking_test", {}).get("recall@10", 0), 4),
        "catalog_coverage": round(metrics.get("ranking_test", {}).get("catalog_coverage", 0), 4),
        "model_backend": meta.get("ctr_backend", "unknown"),
    }
