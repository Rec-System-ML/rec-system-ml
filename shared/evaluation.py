from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def compute_classification_metrics(
    y_true: Iterable[float], y_score: Iterable[float], threshold: float = 0.5
) -> Dict[str, float]:
    """Compute common binary classification metrics."""
    y_true_arr = np.asarray(list(y_true))
    y_score_arr = np.asarray(list(y_score))
    y_pred = (y_score_arr >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true_arr, y_score_arr))
    except ValueError:
        metrics["auc"] = float("nan")

    try:
        metrics["logloss"] = float(log_loss(y_true_arr, y_score_arr, labels=[0, 1]))
    except ValueError:
        metrics["logloss"] = float("nan")
    return metrics


def _topk_by_user(
    pred_df: pd.DataFrame,
    user_col: str,
    score_col: str,
    k: int,
) -> pd.DataFrame:
    return (
        pred_df.sort_values([user_col, score_col], ascending=[True, False])
        .groupby(user_col, group_keys=False)
        .head(k)
    )


def _dcg(relevances: List[int]) -> float:
    return float(sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances)))


def evaluate_ranking_local(
    pred_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "video_id",
    score_col: str = "prediction",
    label_col: str = "label",
    k: int = 10,
) -> Dict[str, float]:
    """Local ranking metrics fallback (Precision@K, Recall@K, NDCG@K, Coverage)."""
    if label_col not in truth_df.columns:
        for candidate in ("rating", "is_click", "label"):
            if candidate in truth_df.columns:
                label_col = candidate
                break
    topk = _topk_by_user(pred_df, user_col=user_col, score_col=score_col, k=k)
    truth_pos = truth_df[truth_df[label_col] > 0][[user_col, item_col]].copy()

    merged = topk.merge(truth_pos.assign(relevant=1), on=[user_col, item_col], how="left")
    merged["relevant"] = merged["relevant"].fillna(0).astype(int)

    # Precision@K
    precision_by_user = merged.groupby(user_col)["relevant"].mean()
    precision_at_k = float(precision_by_user.mean()) if len(precision_by_user) else 0.0

    # Recall@K
    relevant_counts = truth_pos.groupby(user_col)[item_col].nunique()
    hit_counts = merged.groupby(user_col)["relevant"].sum()
    aligned_users = sorted(set(relevant_counts.index).union(hit_counts.index))
    recalls: List[float] = []
    for u in aligned_users:
        rel = int(relevant_counts.get(u, 0))
        if rel == 0:
            continue
        hits = int(hit_counts.get(u, 0))
        recalls.append(hits / rel)
    recall_at_k = float(np.mean(recalls)) if recalls else 0.0

    # NDCG@K
    ndcgs: List[float] = []
    for user_id, group in merged.groupby(user_col):
        rels = group.sort_values(score_col, ascending=False)["relevant"].tolist()
        dcg = _dcg(rels)
        ideal_rels = sorted(rels, reverse=True)
        idcg = _dcg(ideal_rels)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    ndcg_at_k = float(np.mean(ndcgs)) if ndcgs else 0.0

    # Coverage
    recommended_items = topk[item_col].nunique()
    all_items = pred_df[item_col].nunique()
    coverage = float(recommended_items / all_items) if all_items else 0.0

    return {
        f"precision@{k}": precision_at_k,
        f"recall@{k}": recall_at_k,
        f"ndcg@{k}": ndcg_at_k,
        "catalog_coverage": coverage,
    }


def evaluate_ranking(
    pred_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "video_id",
    score_col: str = "prediction",
    label_col: str = "label",
    k: int = 10,
    prefer_recommenders_backend: bool = True,
) -> Dict[str, float]:
    """
    Unified ranking evaluation.

    If Microsoft Recommenders is available, this function tries that backend first.
    It falls back to local implementations when dependency/signature mismatches occur.
    """
    if prefer_recommenders_backend:
        try:
            from recommenders.evaluation.python_evaluation import (
                ndcg_at_k,
                precision_at_k,
                recall_at_k,
            )

            # Recommenders expects specific column names by default.
            rating_true = truth_df.rename(
                columns={user_col: "userID", item_col: "itemID", label_col: "rating"}
            )
            rating_pred = pred_df.rename(
                columns={user_col: "userID", item_col: "itemID", score_col: "prediction"}
            )

            metrics = {
                f"precision@{k}": float(
                    precision_at_k(rating_true, rating_pred, col_prediction="prediction", k=k)
                ),
                f"recall@{k}": float(
                    recall_at_k(rating_true, rating_pred, col_prediction="prediction", k=k)
                ),
                f"ndcg@{k}": float(
                    ndcg_at_k(rating_true, rating_pred, col_prediction="prediction", k=k)
                ),
            }
            # Coverage is computed locally to avoid version differences.
            local_cov = evaluate_ranking_local(
                pred_df=pred_df,
                truth_df=truth_df,
                user_col=user_col,
                item_col=item_col,
                score_col=score_col,
                label_col=label_col,
                k=k,
            )
            metrics["catalog_coverage"] = local_cov["catalog_coverage"]
            return metrics
        except Exception:
            # Fall back to local, keeping pipeline robust for coursework.
            pass

    return evaluate_ranking_local(
        pred_df=pred_df,
        truth_df=truth_df,
        user_col=user_col,
        item_col=item_col,
        score_col=score_col,
        label_col=label_col,
        k=k,
    )


def calculate_coverage(recommendations: Sequence[Sequence[int]], total_items: int) -> float:
    """Compute catalog coverage from user recommendation lists."""
    if total_items <= 0:
        return 0.0
    recommended_items = set()
    for user_recs in recommendations:
        recommended_items.update(user_recs)
    return float(len(recommended_items) / total_items)


def calculate_gini_coefficient(item_exposure_counts: Iterable[float]) -> float:
    """Compute Gini coefficient over item exposure counts."""
    arr = np.asarray(list(item_exposure_counts), dtype=float)
    if arr.size == 0 or np.sum(arr) <= 0:
        return 0.0
    sorted_counts = np.sort(arr)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n)


def evaluate_model(
    predictions: Iterable[float],
    ground_truth: Iterable[float],
    threshold: float = 0.5,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Simple helper used by CDS525 training scripts.

    Returns classification metrics and merges user-provided extra metrics.
    """
    metrics = compute_classification_metrics(
        y_true=ground_truth,
        y_score=predictions,
        threshold=threshold,
    )
    if extra_metrics:
        metrics.update({k: float(v) for k, v in extra_metrics.items()})
    return metrics
