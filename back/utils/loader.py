"""
utils/loader.py
---------------
单例：启动时加载 artifact + 原始数据，供所有 router 共用。
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .pipeline import parse_tag_ids


# ── 全局状态 ────────────────────────────────────────────────────────────────

artifact: dict | None = None          # joblib artifact
user_tag_profiles: dict | None = None # {user_id: {tag_id: weight, ...}}
video_tags: dict | None = None        # {video_id: [tag_id, ...]}
data_dir: Path | None = None
popularity_ranking: list[tuple[int, int]] = []  # [(video_id, click_count), ...] 热门排序


# ── Tag 显示名 ───────────────────────────────────────────────────────────────

def tag_name(tag_id: int) -> str:
    return f"Tag-{tag_id}"


# ── 加载函数（由 lifespan 调用） ─────────────────────────────────────────────

def load_all(artifact_path: Path, raw_data_dir: Path) -> None:
    global artifact, user_tag_profiles, video_tags, data_dir, popularity_ranking

    data_dir = raw_data_dir

    # 1. 模型 artifact
    artifact = joblib.load(artifact_path)

    # 2. 视频基础特征 → tag 映射
    basic_path = raw_data_dir / "video_features_basic_1k.csv"
    basic_df = pd.read_csv(basic_path, usecols=["video_id", "tag"])
    basic_df["tag_list"] = basic_df["tag"].apply(parse_tag_ids)
    video_tags = {
        int(row.video_id): row.tag_list
        for row in basic_df.itertuples()
    }

    # 3. 交互日志 → 用户 tag 画像 + 全局热门排行
    log_path = raw_data_dir / "log_standard_4_22_to_5_08_1k.csv"
    log_df = pd.read_csv(
        log_path,
        usecols=["user_id", "video_id", "is_click", "long_view"],
        dtype={"user_id": int, "video_id": int, "is_click": int},
    )
    log_df["long_view"] = pd.to_numeric(log_df["long_view"], errors="coerce").fillna(0)
    log_df["weight"] = log_df["is_click"].astype(float) + log_df["long_view"].clip(0, 1).astype(float)
    clicked = log_df[log_df["weight"] > 0]

    # 全局热门排行（按点击数）
    pop = clicked.groupby("video_id")["is_click"].sum().sort_values(ascending=False)
    popularity_ranking = [(int(vid), int(cnt)) for vid, cnt in pop.items()]

    # 用户 tag 画像
    profiles: dict[int, dict[int, float]] = {}
    for row in clicked.itertuples():
        uid = int(row.user_id)
        tags = video_tags.get(int(row.video_id), [])
        if not tags:
            continue
        if uid not in profiles:
            profiles[uid] = {}
        for t in tags:
            profiles[uid][t] = profiles[uid].get(t, 0.0) + row.weight

    for uid, tag_weights in profiles.items():
        total = sum(tag_weights.values())
        profiles[uid] = {t: round(w / total * 100, 1) for t, w in tag_weights.items()}

    user_tag_profiles = profiles


# ── 对外工具函数 ─────────────────────────────────────────────────────────────

def get_artifact() -> dict:
    if artifact is None:
        raise RuntimeError("Artifact not loaded.")
    return artifact


def get_user_tag_profile(user_id: int) -> list[dict]:
    """返回 [{tag_id, name, pct}, ...] 按 pct 降序，最多8个。"""
    if user_tag_profiles is None:
        return []
    raw = user_tag_profiles.get(int(user_id), {})
    sorted_tags = sorted(raw.items(), key=lambda x: x[1], reverse=True)[:8]
    return [{"tag_id": t, "name": tag_name(t), "pct": pct} for t, pct in sorted_tags]


def get_video_tags(video_id: int) -> list[int]:
    if video_tags is None:
        return []
    return video_tags.get(int(video_id), [])


def get_popular_candidates(n: int = 500, exclude: set[int] | None = None) -> list[int]:
    """返回全局热门视频 ID 列表，排除已看过的。"""
    exc = exclude or set()
    return [vid for vid, _ in popularity_ranking if vid not in exc][:n]


def knn_score_from_history(clicked_videos: list[int], candidate_video: int) -> float:
    """
    根据点击历史（虚拟用户）给候选视频打 KNN 分。
    复用 ItemKNN 的邻居得分表，不需要重新训练。
    """
    art = get_artifact()
    item_knn = art["item_knn"]
    history_set = set(clicked_videos)
    sim_sum = 0.0
    for hist_item in history_set:
        for neigh_item, sim in item_knn.item_neighbor_scores.get(hist_item, []):
            if neigh_item == candidate_video:
                sim_sum += sim
    return sim_sum
