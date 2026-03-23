from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# Module 1 後處理：時間衰減重排器（TimeDecayReranker）
#
# 職責：在 ItemKNN + XGBoost 混合打分（blend_score）之後，
# 疊加「候選視頻本身的新舊程度」作為第三層加權，
# 讓推薦結果在模型分相近時優先呈現較新的視頻。
#
# 注意：這裡的「時間衰減」與 Module 2（graph_builder.py）的衰減完全不同：
#   Module 2 decay  = exp(−λ × Δdays)：衡量「用戶對某話題的興趣新鮮度」
#   TimeDecayReranker：衡量「候選視頻本身發布/互動時間的新舊排名」
#
# 最終打分公式：
#   final_score = γ × blend_score + (1 − γ) × recency_score
#   其中 γ = 0.75（75% 看模型分，25% 看視頻新舊）
#
# recency_score 的計算：
#   1. 將所有候選視頻按時間戳從新到舊排名（rank = 1, 2, 3, ...）
#   2. recency_score(rank) = 1 / rank^β（β=1.0 即倒數排名）
#   3. 歸一化到 [0, 1]
#   → 最新的視頻 rank=1 得分最高（1.0），越舊得分越低
# ═══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class TimeDecayReranker:
    """Module 1 推薦結果的時間衰減重排器。

    在混合模型分（blend_score = 0.6×XGBoost + 0.4×KNN）基礎上，
    疊加候選視頻的時間新舊排名，生成最終 rerank_score。

    Attributes
    ----------
    gamma : float
        模型分的權重，1−gamma 為時間新舊排名的權重。
        默認 0.75（在 main.py 中調用時傳入 gamma=0.75）。
    beta : float
        排名衰減指數，recency_score(rank) = 1/rank^beta。
        beta=1.0 為倒數衰減（線性懲罰越舊的視頻）。
        beta>1 懲罰更陡，beta<1 懲罰更緩。
    """

    gamma: float = 0.7   # 模型分比重（調用時可覆蓋為 0.75）
    beta:  float = 1.0   # 排名衰減指數

    def __post_init__(self) -> None:
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")
        if self.beta <= 0:
            raise ValueError("beta must be > 0")

    def rerank(
        self,
        pred_scores: np.ndarray,
        timestamps:  np.ndarray,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """對候選集重排，返回 Top-K 的索引和最終分數。

        Parameters
        ----------
        pred_scores : np.ndarray
            每個候選視頻的混合模型分（blend_score），形狀 (N,)。
        timestamps : np.ndarray
            每個候選視頻的時間戳，形狀 (N,)。值越大越新。
        top_k : int
            返回分數最高的前 K 個視頻索引。

        Returns
        -------
        indices : np.ndarray
            Top-K 候選在原始數組中的索引（按 final_score 降序）。
        scores : np.ndarray
            對應的 final_score 值。
        """
        if len(pred_scores) != len(timestamps):
            raise ValueError("pred_scores and timestamps must have same length")
        if len(pred_scores) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        n = len(pred_scores)
        s_pred = np.asarray(pred_scores, dtype=float)

        # ── 計算 recency_score ────────────────────────────────────────────
        # 將候選視頻按時間戳從新到舊排列（argsort 降序），賦予排名 1, 2, 3, ...
        # recency_score(rank) = 1 / rank^beta
        #   rank=1（最新）→ 1.0，rank=2 → 0.5，rank=3 → 0.33，...（beta=1.0 時）
        order = np.argsort(-np.asarray(timestamps))   # 時間戳降序排列的索引
        s_recency = np.zeros(n, dtype=float)
        for rank, idx in enumerate(order, start=1):
            s_recency[idx] = 1.0 / (rank ** self.beta)

        # 歸一化到 [0, 1]，使 recency_score 與 pred_scores 的量綱可比
        if s_recency.max() > 0:
            s_recency = s_recency / s_recency.max()

        # ── 加權合併：final_score = γ × 模型分 + (1−γ) × 時間新舊分 ────
        final_scores = self.gamma * s_pred + (1.0 - self.gamma) * s_recency

        # 取 Top-K
        indices = np.argsort(-final_scores)[:top_k]
        return indices, final_scores[indices]

    def rerank_dataframe(
        self,
        df: pd.DataFrame,
        score_col: str = "prediction",
        time_col:  str = "timestamp",
        top_k: int = 10,
    ) -> pd.DataFrame:
        """DataFrame 接口：輸入候選 DataFrame，返回 Top-K 重排結果。

        在原始 DataFrame 上追加 ``rerank_score`` 列，按降序排列返回前 K 行。

        Parameters
        ----------
        df : pd.DataFrame
            候選視頻 DataFrame，至少包含 score_col 和 time_col 兩列。
        score_col : str
            模型混合分列名（默認 "prediction"，即 blend_score）。
        time_col : str
            時間戳列名（默認 "timestamp"；demo_app 中使用 "time_ms"）。
        top_k : int
            返回前 K 行。
        """
        indices, scores = self.rerank(
            pred_scores=df[score_col].to_numpy(),
            timestamps=df[time_col].to_numpy(),
            top_k=top_k,
        )
        out = df.iloc[indices].copy()
        out["rerank_score"] = scores
        return out.sort_values("rerank_score", ascending=False).reset_index(drop=True)
