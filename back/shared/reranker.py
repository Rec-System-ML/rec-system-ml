from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class TimeDecayReranker:
    """
    Score = gamma * prediction + (1 - gamma) * recency_score
    recency_score(rank) = 1 / rank^beta, where rank is by timestamp desc.
    """

    gamma: float = 0.7
    beta: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")
        if self.beta <= 0:
            raise ValueError("beta must be > 0")

    def rerank(
        self,
        pred_scores: np.ndarray,
        timestamps: np.ndarray,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(pred_scores) != len(timestamps):
            raise ValueError("pred_scores and timestamps must have same length")
        if len(pred_scores) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        n = len(pred_scores)
        s_pred = np.asarray(pred_scores, dtype=float)

        order = np.argsort(-np.asarray(timestamps))
        s_recency = np.zeros(n, dtype=float)
        for rank, idx in enumerate(order, start=1):
            s_recency[idx] = 1.0 / (rank ** self.beta)
        if s_recency.max() > 0:
            s_recency = s_recency / s_recency.max()

        final_scores = self.gamma * s_pred + (1.0 - self.gamma) * s_recency
        indices = np.argsort(-final_scores)[:top_k]
        return indices, final_scores[indices]

    def rerank_dataframe(
        self,
        df: pd.DataFrame,
        score_col: str = "prediction",
        time_col: str = "timestamp",
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Return top-k rows sorted by reranked score."""
        indices, scores = self.rerank(
            pred_scores=df[score_col].to_numpy(),
            timestamps=df[time_col].to_numpy(),
            top_k=top_k,
        )
        out = df.iloc[indices].copy()
        out["rerank_score"] = scores
        return out.sort_values("rerank_score", ascending=False).reset_index(drop=True)
