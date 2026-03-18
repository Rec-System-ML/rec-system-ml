from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass
class ItemKNNConfig:
    n_neighbors: int = 50
    metric: str = "cosine"
    positive_threshold: int = 1


class ItemKNNRecommender:
    def __init__(self, config: ItemKNNConfig | None = None) -> None:
        self.config = config or ItemKNNConfig()
        self._fitted = False

        self.user_histories: Dict[int, Set[int]] = {}
        self.item_neighbor_scores: Dict[int, List[Tuple[int, float]]] = {}
        self.item_click_popularity: Dict[int, int] = {}

    def fit(self, train_df: pd.DataFrame) -> "ItemKNNRecommender":
        required_cols = {"user_id", "video_id", "is_click"}
        missing = required_cols - set(train_df.columns)
        if missing:
            raise KeyError(f"train_df missing required columns: {missing}")

        pos = train_df[train_df["is_click"] >= self.config.positive_threshold].copy()
        if pos.empty:
            raise ValueError("No positive interactions found for ItemKNN training.")

        user_item = (
            pos.assign(value=1)
            .pivot_table(
                index="video_id",
                columns="user_id",
                values="value",
                aggfunc="max",
                fill_value=0,
            )
            .astype(np.float32)
        )

        item_ids = user_item.index.to_numpy()
        matrix = user_item.to_numpy()
        max_neighbors = min(self.config.n_neighbors + 1, len(item_ids))

        nn = NearestNeighbors(n_neighbors=max_neighbors, metric=self.config.metric)
        nn.fit(matrix)
        distances, indices = nn.kneighbors(matrix)

        item_neighbor_scores: Dict[int, List[Tuple[int, float]]] = {}
        for row_idx, item_id in enumerate(item_ids):
            neighbors: List[Tuple[int, float]] = []
            for col_idx, distance in zip(indices[row_idx], distances[row_idx]):
                neighbor_item = int(item_ids[col_idx])
                if neighbor_item == int(item_id):
                    continue
                similarity = float(1.0 - distance)
                neighbors.append((neighbor_item, similarity))
            item_neighbor_scores[int(item_id)] = neighbors

        user_histories = (
            pos.groupby("user_id")["video_id"]
            .apply(lambda x: set(int(v) for v in x.tolist()))
            .to_dict()
        )
        popularity = pos["video_id"].value_counts().astype(int).to_dict()

        self.user_histories = {int(k): v for k, v in user_histories.items()}
        self.item_neighbor_scores = item_neighbor_scores
        self.item_click_popularity = {int(k): int(v) for k, v in popularity.items()}
        self._fitted = True
        return self

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("ItemKNNRecommender is not fitted.")

    def score(self, user_id: int, item_id: int) -> float:
        self._require_fit()
        history = self.user_histories.get(int(user_id), set())
        if not history:
            pop = self.item_click_popularity.get(int(item_id), 0)
            return float(pop)

        sim_sum = 0.0
        for hist_item in history:
            for neigh_item, sim in self.item_neighbor_scores.get(hist_item, []):
                if neigh_item == int(item_id):
                    sim_sum += sim
        return float(sim_sum)

    def recommend(
        self,
        user_id: int,
        candidate_items: Sequence[int],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        self._require_fit()
        seen = self.user_histories.get(int(user_id), set())
        scores = []
        for item in candidate_items:
            item_int = int(item)
            if item_int in seen:
                continue
            scores.append((item_int, self.score(user_id=user_id, item_id=item_int)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def explain(self, user_id: int, item_id: int, top_n: int = 3) -> str:
        self._require_fit()
        history = self.user_histories.get(int(user_id), set())
        if not history:
            return "新用户/低活跃：按整体热门度推荐。"

        evidence: List[Tuple[int, float]] = []
        for hist_item in history:
            for neigh_item, sim in self.item_neighbor_scores.get(hist_item, []):
                if neigh_item == int(item_id):
                    evidence.append((hist_item, sim))

        if not evidence:
            return "该视频与历史偏好相近度有限，作为探索推荐。"

        evidence.sort(key=lambda x: x[1], reverse=True)
        top = evidence[:top_n]
        msg = ", ".join([f"你看过 {hid}（相似度 {sim:.3f}）" for hid, sim in top])
        return f"因为 {msg}"

    def candidate_pool(self, user_id: int, top_pop_n: int = 500) -> List[int]:
        self._require_fit()
        seen = self.user_histories.get(int(user_id), set())
        candidates: Set[int] = set()

        for hist_item in seen:
            for neigh_item, _sim in self.item_neighbor_scores.get(hist_item, []):
                if neigh_item not in seen:
                    candidates.add(neigh_item)

        popular_items = [item for item, _cnt in sorted(self.item_click_popularity.items(), key=lambda x: x[1], reverse=True)]
        for item in popular_items[:top_pop_n]:
            if item not in seen:
                candidates.add(item)

        return list(candidates)
