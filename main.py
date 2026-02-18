from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models import CTRFeatureBuilder, CTRModel, ItemKNNConfig, ItemKNNRecommender
from shared_bootstrap import ensure_shared_on_path


def _load_interactions(data_dir: Path, nrows: int) -> pd.DataFrame:
    interactions_file = data_dir / "log_standard_4_22_to_5_08_1k.csv"
    usecols = ["user_id", "video_id", "is_click", "time_ms", "date"]
    df = pd.read_csv(interactions_file, usecols=usecols, nrows=nrows)
    df = df.dropna(subset=["user_id", "video_id", "is_click"]).copy()
    df["user_id"] = df["user_id"].astype(int)
    df["video_id"] = df["video_id"].astype(int)
    df["is_click"] = df["is_click"].astype(int)
    return df


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("time_ms").reset_index(drop=True)
    n = len(df)
    n_train = int(0.7 * n)
    n_val = int(0.1 * n)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    return train, val, test


def _ranking_metrics(pred_df: pd.DataFrame, truth_df: pd.DataFrame, k: int = 10) -> dict:
    from evaluation import evaluate_ranking

    return evaluate_ranking(
        pred_df=pred_df,
        truth_df=truth_df.rename(columns={"is_click": "label"}),
        user_col="user_id",
        item_col="video_id",
        score_col="prediction",
        label_col="label",
        k=k,
    )


def train_and_evaluate(sample_rows: int, data_dir: Path, output_dir: Path) -> dict:
    from evaluation import compute_classification_metrics
    from reranker import TimeDecayReranker

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    artifacts_dir = output_dir / "artifacts"
    checkpoints_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)

    df = _load_interactions(data_dir=data_dir, nrows=sample_rows)
    train_df, val_df, test_df = _temporal_split(df)

    # Model 1: ItemKNN
    item_knn = ItemKNNRecommender(ItemKNNConfig(n_neighbors=60))
    item_knn.fit(train_df)

    val_knn_scores = np.array([item_knn.score(u, i) for u, i in val_df[["user_id", "video_id"]].to_numpy()])
    test_knn_scores = np.array([item_knn.score(u, i) for u, i in test_df[["user_id", "video_id"]].to_numpy()])

    val_knn_prob = 1.0 / (1.0 + np.exp(-val_knn_scores))
    test_knn_prob = 1.0 / (1.0 + np.exp(-test_knn_scores))

    # Model 2: XGBoost CTR
    feat_builder = CTRFeatureBuilder().fit(train_df)
    X_train = feat_builder.transform(train_df)
    X_val = feat_builder.transform(val_df)
    X_test = feat_builder.transform(test_df)
    y_train = train_df["is_click"]

    ctr_model = CTRModel(random_state=42).fit(X_train, y_train)
    val_xgb_prob = ctr_model.predict_proba(X_val)
    test_xgb_prob = ctr_model.predict_proba(X_test)

    # Blend score for ranking and demo
    val_blend = 0.6 * val_xgb_prob + 0.4 * val_knn_prob
    test_blend = 0.6 * test_xgb_prob + 0.4 * test_knn_prob

    # Time-decay rerank score (used later in demo; keep as optional component)
    reranker = TimeDecayReranker(gamma=0.75, beta=1.0)
    _ = reranker  # keep object persisted in artifact

    metrics = {
        "item_knn_val": compute_classification_metrics(val_df["is_click"], val_knn_prob),
        "item_knn_test": compute_classification_metrics(test_df["is_click"], test_knn_prob),
        "ctr_val": compute_classification_metrics(val_df["is_click"], val_xgb_prob),
        "ctr_test": compute_classification_metrics(test_df["is_click"], test_xgb_prob),
        "blend_val": compute_classification_metrics(val_df["is_click"], val_blend),
        "blend_test": compute_classification_metrics(test_df["is_click"], test_blend),
    }

    pred_test = test_df[["user_id", "video_id"]].copy()
    pred_test["prediction"] = test_blend
    metrics["ranking_test"] = _ranking_metrics(pred_df=pred_test, truth_df=test_df, k=10)

    artifact = {
        "item_knn": item_knn,
        "ctr_model": ctr_model,
        "feature_builder": feat_builder,
        "reranker": reranker,
        "metrics": metrics,
        "train_users": sorted(train_df["user_id"].unique().tolist()),
        "train_items": sorted(train_df["video_id"].unique().tolist()),
        "data_meta": {
            "rows_used": int(len(df)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "data_dir": str(data_dir),
            "ctr_backend": ctr_model.model_name,
        },
    }

    joblib.dump(artifact, checkpoints_dir / "mvp_artifact.joblib")
    with (artifacts_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CDS524 MVP recommender.")
    parser.add_argument("--rows", type=int, default=250_000, help="Rows loaded from KuaiRand log csv")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Override dataset directory. Default points to recsys-shared KuaiRand-1K data folder.",
    )
    parser.add_argument("--output-dir", type=str, default=".", help="Project output root")
    return parser.parse_args()


if __name__ == "__main__":
    shared_path = ensure_shared_on_path()
    args = parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
    else:
        data_dir = (shared_path / "data/KuaiRand-1K/data").resolve()

    output_dir = Path(args.output_dir).expanduser().resolve()
    metrics = train_and_evaluate(sample_rows=args.rows, data_dir=data_dir, output_dir=output_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
