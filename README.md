# KuaiRand-1K Recommendation System Dashboard

A recommendation system built on the KuaiRand-1K dataset, covering three modules: cold-start personalization, real-time re-ranking for returning users, and interest graph evolution. Includes a complete backend API and an interactive frontend dashboard.

---

## Modules

| Module | Description |
|--------|-------------|
| **Overview Dashboard** | KPI stats, CTR signal comparison, three-version model metrics (Baseline / +ItemKNN / +XGBoost) |
| **User Profile** | User activity, historical interest tag distribution (donut chart), KNN neighbor positioning |
| **Live Simulate** | New-user cold-start (four-phase progressive personalization) + returning-user real-time re-ranking |
| **Interest Graph** | D3.js force-directed graph showing Markov transitions and time-decay evolution of user interest nodes |

---

## Tech Stack

**Backend**
- FastAPI + Uvicorn (port 8100)
- ItemKNN collaborative filtering + XGBoost CTR prediction (blended score 0.4:0.6)
- Time-decay re-ranking (`w(t) = exp(-λΔt)`)
- K-Means user clustering for cold-start cluster matching (K=6)

**Frontend**
- Pure HTML / CSS / JS (no framework dependencies)
- D3.js v7 force-directed graph
- Chart.js for statistical charts

---

## Dataset

Uses the [KuaiRand-1K](https://kuairand.com/) dataset. Download it manually and place it under `KuaiRand-1K/data/`:

```
KuaiRand-1K/data/
├── log_standard_4_08_to_4_21_1k.csv   # Phase 1 training log
├── log_standard_4_22_to_5_08_1k.csv   # Phase 2 training log
├── log_random_4_22_to_5_08_1k.csv     # Random exposure (de-bias evaluation)
├── user_features_1k.csv
├── video_features_basic_1k.csv
└── video_features_statistic_1k.csv
```

---

## Quick Start

### 1. Install dependencies

```bash
cd back
pip install -r requirements.txt
```

### 2. Start the backend

```bash
cd back
python api.py
```

Open the dashboard at [http://localhost:8100](http://localhost:8100).
API docs: [http://localhost:8100/docs](http://localhost:8100/docs)

> **Pre-trained model included** (`back/checkpoints/mvp_artifact.joblib`). You can start directly without retraining.

### 3. (Optional) Retrain the model

```bash
cd back
python train.py --data-dir ../KuaiRand-1K/data --rows 250000
```

Trains ItemKNN + XGBoost and saves the artifact to `back/checkpoints/mvp_artifact.joblib`.

---

## Project Structure

```
.
├── back/
│   ├── api.py                   # FastAPI entry point (with static file mount)
│   ├── train.py                 # Model training script
│   ├── reranker.py              # Compatibility shim for joblib deserialization
│   ├── checkpoints/
│   │   └── mvp_artifact.joblib  # Pre-trained model artifact
│   ├── artifacts/
│   │   └── metrics.json         # Per-model evaluation metrics
│   ├── models/
│   │   ├── item_knn.py          # ItemKNN collaborative filtering
│   │   ├── xgboost_ctr.py       # XGBoost CTR model
│   │   └── reranker.py          # Time-decay re-ranker
│   ├── routers/
│   │   ├── stats.py             # Stats & metrics API
│   │   ├── user.py              # User profile API
│   │   ├── recommend.py         # Recommendation & real-time re-ranking API
│   │   ├── cold_start.py        # Cold-start API (four-phase progressive strategy)
│   │   └── interest_graph.py    # Interest graph API
│   └── utils/
│       ├── loader.py            # Artifact loading & K-Means cluster pre-computation
│       ├── graph_builder.py     # Interest graph construction (time-decay + Markov transitions)
│       ├── pipeline.py          # Data loading and preprocessing
│       ├── evaluation.py        # Evaluation metric computation
│       └── tags.py              # Tag ID → display name mapping
├── ui_prototype/
│   ├── index.html               # Main dashboard
│   ├── style.css                # Global styles
│   └── app.js                   # Interaction logic & charts
└── KuaiRand-1K/                 # Dataset directory (download data/ manually)
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/api/stats` | Model evaluation metrics (AUC, NDCG@10, etc.) |
| GET  | `/api/users` | Available user list |
| GET  | `/api/user/{id}/profile` | User profile (tag distribution, activity) |
| GET  | `/api/popular` | Global popular videos, Top-N |
| POST | `/api/recommend` | Returning-user recommendation (ItemKNN + XGBoost blend) |
| POST | `/api/recommend/realtime` | Returning-user real-time re-ranking (with session clicks) |
| POST | `/api/cold-start/recommend` | New-user cold-start recommendation (four-phase progressive) |
| GET  | `/api/interest-graph/{user_id}` | User interest evolution graph data |
| GET  | `/api/interest-graph/demo` | Built-in demo graph (no data required) |

---

## Cold-Start Strategy

New-user cold-start uses a four-phase progressive strategy, gradually increasing personalization as clicks accumulate:

| Phase | Clicks | Strategy |
|-------|--------|----------|
| Phase 0 | 0 | Pure global popularity fallback |
| Phase 1 | 1–7 | Popular × (1−eff_conf) + Cluster × eff_conf; confidence grows linearly with clicks |
| Phase 2 | 8–19 | Cluster candidates dominate; KNN weight linearly ramps from 0% to 40% |
| Phase 3 | 20+ | Cluster ∪ KNN neighbor candidate fusion; KNN weight continues ramping to 60% |

Clusters are built via K-Means (K=6) on existing-user tag profiles. After the new user clicks, their tag vector is compared to cluster centroids using cosine similarity for assignment.

---

## Dataset License

KuaiRand-1K is subject to its original license. See `KuaiRand-1K/LICENSE`.
