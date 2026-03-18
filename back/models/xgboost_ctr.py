from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

from sklearn.ensemble import RandomForestClassifier


@dataclass
class FeatureStats:
    user_ctr: Dict[int, float]
    item_ctr: Dict[int, float]
    user_click_count: Dict[int, int]
    item_click_count: Dict[int, int]
    global_ctr: float


class CTRFeatureBuilder:
    def __init__(self) -> None:
        self.stats: FeatureStats | None = None

    def fit(self, train_df: pd.DataFrame) -> "CTRFeatureBuilder":
        if train_df.empty:
            raise ValueError("train_df is empty.")
        user_ctr = train_df.groupby("user_id")["is_click"].mean().to_dict()
        item_ctr = train_df.groupby("video_id")["is_click"].mean().to_dict()
        user_click_count = train_df.groupby("user_id")["is_click"].sum().astype(int).to_dict()
        item_click_count = train_df.groupby("video_id")["is_click"].sum().astype(int).to_dict()
        global_ctr = float(train_df["is_click"].mean())
        self.stats = FeatureStats(
            user_ctr={int(k): float(v) for k, v in user_ctr.items()},
            item_ctr={int(k): float(v) for k, v in item_ctr.items()},
            user_click_count={int(k): int(v) for k, v in user_click_count.items()},
            item_click_count={int(k): int(v) for k, v in item_click_count.items()},
            global_ctr=global_ctr,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.stats is None:
            raise RuntimeError("CTRFeatureBuilder is not fitted.")
        out = df.copy()
        out["user_ctr"] = out["user_id"].map(self.stats.user_ctr).fillna(self.stats.global_ctr)
        out["item_ctr"] = out["video_id"].map(self.stats.item_ctr).fillna(self.stats.global_ctr)
        out["user_click_count"] = out["user_id"].map(self.stats.user_click_count).fillna(0)
        out["item_click_count"] = out["video_id"].map(self.stats.item_click_count).fillna(0)
        out["hour"] = ((out.get("time_ms", 0) // (1000 * 60 * 60)) % 24).astype(int)
        out["user_item_ctr_gap"] = out["user_ctr"] - out["item_ctr"]
        return out[
            [
                "user_ctr",
                "item_ctr",
                "user_click_count",
                "item_click_count",
                "hour",
                "user_item_ctr_gap",
            ]
        ].astype(float)


class CTRModel:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model = None
        self.model_name = ""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CTRModel":
        if XGBClassifier is not None:
            self.model = XGBClassifier(
                n_estimators=120,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=4,
            )
            self.model_name = "xgboost"
        else:
            self.model = RandomForestClassifier(
                n_estimators=220,
                max_depth=12,
                random_state=self.random_state,
                n_jobs=4,
            )
            self.model_name = "random_forest_fallback"
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CTRModel is not fitted.")
        return self.model.predict_proba(X)[:, 1]
