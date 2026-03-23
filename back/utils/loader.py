"""
utils/loader.py
---------------
Singleton module: loads the artifact and raw data at startup for use by all routers.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .pipeline import parse_tag_ids


artifact: dict | None = None           # joblib artifact
user_tag_profiles: dict | None = None  # {user_id: {tag_id: weight, ...}}
video_tags: dict | None = None         # {video_id: [tag_id, ...]}
data_dir: Path | None = None
popularity_ranking: list[tuple[int, int]] = []   # [(video_id, click_count), ...]
user_register_days: dict[int, int] = {}          # {user_id: register_days}

tag_matrix: "pd.DataFrame | None" = None
interactions_df: "pd.DataFrame | None" = None

cluster_centers: "np.ndarray | None" = None   # shape (K, n_tags)
cluster_items: dict[int, list[int]] = {}       # {cluster_id: [video_id, ...]}
cluster_tag_index: dict[int, int] = {}         # {tag_id: column_index}
N_CLUSTERS = 6


def tag_name(tag_id: int) -> str:
    return f"Tag-{tag_id}"


def _build_clusters(profiles: dict[int, dict[int, float]],
                    item_knn_histories: dict[int, set[int]]) -> None:
    """
    K-Means clustering on all user tag profiles.
    cluster_centers: centroids in tag space
    cluster_items:   top videos aggregated from each cluster's members' watch history
    """
    global cluster_centers, cluster_items, cluster_tag_index

    if len(profiles) < N_CLUSTERS:
        print(f"[WARNING] Not enough users ({len(profiles)}) for {N_CLUSTERS} clusters, skipping")
        return

    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("[WARNING] sklearn not installed, skipping clustering")
        return

    all_tags = sorted({t for tags in (video_tags or {}).values() for t in tags})
    tag_to_idx = {t: i for i, t in enumerate(all_tags)}
    cluster_tag_index = tag_to_idx

    user_ids = list(profiles.keys())
    X = np.zeros((len(user_ids), len(all_tags)), dtype=np.float32)
    for row, uid in enumerate(user_ids):
        for tag, weight in profiles[uid].items():
            if tag in tag_to_idx:
                X[row, tag_to_idx[tag]] = weight

    k = min(N_CLUSTERS, len(user_ids) // 2)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    cluster_centers = km.cluster_centers_   # shape (k, n_tags)

    cluster_video_cnt: dict[int, dict[int, int]] = {c: {} for c in range(k)}
    for row, uid in enumerate(user_ids):
        c = int(labels[row])
        for vid in item_knn_histories.get(uid, set()):
            cluster_video_cnt[c][vid] = cluster_video_cnt[c].get(vid, 0) + 1

    cluster_items = {}
    for c, video_cnts in cluster_video_cnt.items():
        sorted_vids = sorted(video_cnts.items(), key=lambda x: x[1], reverse=True)
        cluster_items[c] = [vid for vid, _ in sorted_vids[:300]]

    print(f"[INFO] Cluster-Based: {k} clusters, "
          f"sizes={[int((labels==c).sum()) for c in range(k)]}")


def load_all(artifact_path: Path, raw_data_dir: Path) -> None:
    global artifact, user_tag_profiles, video_tags, data_dir, popularity_ranking
    global tag_matrix, interactions_df, user_register_days

    data_dir = raw_data_dir

    artifact = joblib.load(artifact_path)

    user_feat_path = raw_data_dir / "user_features_1k.csv"
    user_feat_df = pd.read_csv(user_feat_path, usecols=["user_id", "register_days"])
    user_register_days = {int(r.user_id): int(r.register_days) for r in user_feat_df.itertuples()}

    basic_path = raw_data_dir / "video_features_basic_1k.csv"
    basic_df = pd.read_csv(basic_path, usecols=["video_id", "tag"])
    basic_df["tag_list"] = basic_df["tag"].apply(parse_tag_ids)
    video_tags = {
        int(row.video_id): row.tag_list
        for row in basic_df.itertuples()
    }

    log_path = raw_data_dir / "log_standard_4_22_to_5_08_1k.csv"
    _avail_cols = set(pd.read_csv(log_path, nrows=0).columns.tolist())
    _base_cols  = ["user_id", "video_id", "is_click", "long_view"]
    _graph_cols = ["time_ms", "is_like", "is_follow", "is_comment", "is_forward", "play_time_ms"]
    _load_cols  = [c for c in _base_cols + _graph_cols if c in _avail_cols]

    log_df = pd.read_csv(
        log_path,
        usecols=_load_cols,
        dtype={"user_id": int, "video_id": int, "is_click": int},
    )
    log_df["long_view"] = pd.to_numeric(log_df["long_view"], errors="coerce").fillna(0)
    log_df["weight"] = log_df["is_click"].astype(float) + log_df["long_view"].clip(0, 1).astype(float)
    clicked = log_df[log_df["weight"] > 0]

    pop = clicked.groupby("video_id")["is_click"].sum().sort_values(ascending=False)
    popularity_ranking = [(int(vid), int(cnt)) for vid, cnt in pop.items()]

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

    _build_clusters(profiles, artifact["item_knn"].user_histories)

    interactions_df = log_df.head(250_000).copy()
    print(f"[INFO] interactions_df: {len(interactions_df):,} rows, cols={list(interactions_df.columns)}")

    try:
        from sklearn.preprocessing import MultiLabelBinarizer
        _item_ids  = list(video_tags.keys())
        _tag_lists = [video_tags[vid] for vid in _item_ids]
        _mlb = MultiLabelBinarizer()
        _vectors = _mlb.fit_transform(_tag_lists)
        tag_matrix = pd.DataFrame(
            _vectors,
            index=pd.Index(_item_ids, name="video_id"),
            columns=_mlb.classes_.astype(int),
        )
        print(f"[INFO] tag_matrix: {tag_matrix.shape[0]:,} videos × {tag_matrix.shape[1]} tags")
    except Exception as _e:
        print(f"[WARNING] Could not build tag_matrix for interest graph: {_e}")
        tag_matrix = None


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


def get_cluster_candidates(clicked_videos: list[int], n: int = 200,
                            exclude: set[int] | None = None) -> tuple[list[int], int, float]:
    """
    Phase 2: match click history's tag vector to the nearest cluster, return that
    cluster's top videos. Returns (candidates, cluster_id, confidence) where
    confidence = nearest_sim / mean_sim, normalized to [0, 1].
    """
    if cluster_centers is None or not cluster_items:
        exc = set(clicked_videos) | (exclude or set())
        return get_popular_candidates(n=n, exclude=exc), -1, 0.0

    tag_counts = Counter(t for v in clicked_videos for t in get_video_tags(v))
    total = sum(tag_counts.values()) or 1
    vec = np.zeros(len(cluster_tag_index), dtype=np.float32)
    for tag, cnt in tag_counts.items():
        if tag in cluster_tag_index:
            vec[cluster_tag_index[tag]] = cnt / total * 100

    sims = []
    for center in cluster_centers:
        norm = np.linalg.norm(vec) * np.linalg.norm(center)
        sim = float(np.dot(vec, center) / norm) if norm > 1e-9 else 0.0
        sims.append(sim)

    nearest = int(np.argmax(sims))
    mean_sim = float(np.mean(sims)) or 1e-9
    confidence = round(min(sims[nearest] / mean_sim, 3.0) / 3.0, 3)  # normalize to [0,1]

    exc = set(clicked_videos) | (exclude or set())
    cands = [v for v in cluster_items.get(nearest, []) if v not in exc][:n]

    if len(cands) < n:
        extra = get_popular_candidates(n=n - len(cands), exclude=exc | set(cands))
        cands = cands + extra

    return cands, nearest, confidence


def knn_candidates_from_history(clicked_videos: list[int], n: int = 300,
                                exclude: set[int] | None = None) -> list[int]:
    """Phase 3: build candidate pool from KNN neighbors of click history, ranked by cumulative similarity."""
    art = get_artifact()
    item_knn = art["item_knn"]
    seen = set(clicked_videos) | (exclude or set())
    scores: dict[int, float] = {}
    for vid in clicked_videos:
        for neigh, sim in item_knn.item_neighbor_scores.get(vid, []):
            if neigh not in seen:
                scores[neigh] = scores.get(neigh, 0.0) + sim
    sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [vid for vid, _ in sorted_cands[:n]]


def knn_score_from_history(clicked_videos: list[int], candidate_video: int) -> float:
    """Score a candidate video against a virtual user's click history using ItemKNN."""
    art = get_artifact()
    item_knn = art["item_knn"]
    history_set = set(clicked_videos)
    sim_sum = 0.0
    for hist_item in history_set:
        for neigh_item, sim in item_knn.item_neighbor_scores.get(hist_item, []):
            if neigh_item == candidate_video:
                sim_sum += sim
    return sim_sum
