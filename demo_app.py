from __future__ import annotations

from pathlib import Path
from typing import List

import joblib

from shared_bootstrap import ensure_shared_on_path
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def load_artifact(path: str):
    return joblib.load(path)


def build_candidate_frame(user_id: int, candidates: List[int]) -> pd.DataFrame:
    now_ms = pd.Timestamp.now().value // 1_000_000
    return pd.DataFrame(
        {
            "user_id": [int(user_id)] * len(candidates),
            "video_id": [int(v) for v in candidates],
            "time_ms": [int(now_ms)] * len(candidates),
        }
    )


@st.cache_resource(show_spinner="Building interest graph from KuaiRand...")
def _build_graph(user_id: int | None, data_dir: str, sample_rows: int = 250_000):
    from services.graph_builder import build_demo_from_kuairand
    return build_demo_from_kuairand(data_dir=data_dir, user_id=user_id, sample_rows=sample_rows)


def main() -> None:
    shared_path = ensure_shared_on_path()
    st.set_page_config(page_title="CDS524 ML Recommender", layout="wide")
    st.title("CDS524 — KuaiRand Recommendation System")

    tab_rec, tab_graph = st.tabs(["Recommendation", "Interest Evolution Graph"])

    default_path = str((Path(__file__).resolve().parent / "checkpoints/mvp_artifact.joblib").resolve())
    artifact_path = st.sidebar.text_input("Artifact 路径", value=default_path)

    # ── Tab 1: Recommendation ──
    with tab_rec:
        if not Path(artifact_path).exists():
            st.warning("未找到 artifact。先运行: `python main.py --rows 250000`")
            st.stop()

        artifact = load_artifact(artifact_path)
        item_knn = artifact["item_knn"]
        ctr_model = artifact["ctr_model"]
        feature_builder = artifact["feature_builder"]
        reranker = artifact["reranker"]
        metrics = artifact["metrics"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Blend AUC (test)", f"{metrics['blend_test'].get('auc', 0):.4f}")
        c2.metric("NDCG@10", f"{metrics['ranking_test'].get('ndcg@10', 0):.4f}")
        c3.metric("Coverage", f"{metrics['ranking_test'].get('catalog_coverage', 0):.4f}")

        train_users = artifact.get("train_users", [])
        default_user = int(train_users[0]) if train_users else 0
        user_id = st.number_input("输入 user_id", min_value=0, value=default_user, step=1)
        top_k = st.slider("Top-K", min_value=5, max_value=20, value=10, step=1)

        if st.button("生成推荐", type="primary"):
            candidates = item_knn.candidate_pool(user_id=int(user_id), top_pop_n=800)
            if not candidates:
                st.info("该用户候选为空，尝试换一个 user_id。")
                st.stop()

            cand_df = build_candidate_frame(user_id=int(user_id), candidates=candidates)
            cand_df["knn_score"] = cand_df.apply(
                lambda r: item_knn.score(int(r["user_id"]), int(r["video_id"])), axis=1
            )
            X_cand = feature_builder.transform(cand_df)
            cand_df["ctr_score"] = ctr_model.predict_proba(X_cand)
            knn_prob = 1.0 / (1.0 + np.exp(-cand_df["knn_score"]))
            cand_df["prediction"] = 0.6 * cand_df["ctr_score"] + 0.4 * knn_prob

            reranked = reranker.rerank_dataframe(
                cand_df,
                score_col="prediction",
                time_col="time_ms",
                top_k=top_k,
            )

            display = reranked[["video_id", "prediction", "knn_score", "ctr_score", "rerank_score"]].copy()
            display = display.sort_values("rerank_score", ascending=False).reset_index(drop=True)
            display.index = display.index + 1
            st.subheader("Top 推荐结果")
            st.dataframe(display, use_container_width=True)

            st.subheader("推荐解释")
            for _, row in display.head(5).iterrows():
                vid = int(row["video_id"])
                reason = item_knn.explain(int(user_id), vid)
                st.markdown(f"- 视频 `{vid}`: {reason}")

    # ── Tab 2: Interest Evolution Graph ──
    with tab_graph:
        st.subheader("User Interest Evolution Graph")
        st.caption("Powered by D3.js force-directed layout with time-decay encoding & mutation detection")

        data_dir = str((shared_path / "data/KuaiRand-1K/data").resolve())
        graph_user = st.number_input("User ID for graph (0 = 自動選最活躍用戶)", min_value=0, value=0, step=1, key="graph_uid")
        use_demo = st.checkbox("Use built-in demo data (no KuaiRand needed)", value=False)

        if st.button("Generate Graph", type="primary", key="gen_graph"):
            if use_demo:
                from interest_graph import render_demo_interest_graph
                html = render_demo_interest_graph(width=1060, height=680)
                import streamlit.components.v1 as components
                components.html(html, height=720, scrolling=False)
            else:
                try:
                    uid = None if graph_user == 0 else int(graph_user)
                    graph_nodes, graph_links = _build_graph(uid, data_dir)
                    if not graph_nodes:
                        st.info("該用戶無互動數據，改用內建 demo 數據展示。")
                        from interest_graph import render_demo_interest_graph
                        html = render_demo_interest_graph(width=1060, height=680)
                    else:
                        from interest_graph import render_interest_graph
                        html = render_interest_graph(graph_nodes, graph_links, width=1060, height=680)
                    import streamlit.components.v1 as components
                    components.html(html, height=720, scrolling=False)
                except Exception as exc:
                    st.error(f"Graph build failed: {exc}")
                    from interest_graph import render_demo_interest_graph
                    html = render_demo_interest_graph(width=1060, height=680)
                    st.info("Falling back to demo data.")
                    import streamlit.components.v1 as components
                    components.html(html, height=720, scrolling=False)


if __name__ == "__main__":
    main()
