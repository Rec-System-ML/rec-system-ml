"""
routers/interest_graph.py
--------------------------
GET  /api/interest-graph/demo         → built-in demo graph data (no data needed)
GET  /api/interest-graph/{user_id}    → real user interest graph built from KuaiRand data
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

import utils.loader as _loader
from utils.graph_builder import build_interest_graph

router = APIRouter()


# ── Demo data ─────────────────────────────────────────────────────────────────

_DEMO_NODES = [
    {"id":"1",  "label":"Entertainment", "decay":0.92, "status":"active",
     "timestamp":0.10, "tags":[1],  "metrics":{"ENG":"78%","CTR":"45%","WATCH":"62s"}},
    {"id":"2",  "label":"Gaming",        "decay":0.85, "status":"active",
     "timestamp":0.28, "tags":[2],  "metrics":{"ENG":"65%","CTR":"38%","WATCH":"88s"}},
    {"id":"3",  "label":"Food & Cooking","decay":0.71, "status":"active",
     "timestamp":0.45, "tags":[3],  "metrics":{"ENG":"55%","CTR":"32%","WATCH":"45s"}},
    {"id":"4",  "label":"Tech News",     "decay":0.45, "status":"fading",
     "timestamp":0.15, "tags":[4],  "metrics": None},
    {"id":"5",  "label":"Lifestyle",     "decay":0.30, "status":"fading",
     "timestamp":0.08, "tags":[5],  "metrics": None},
    {"id":"6",  "label":"Sports",        "decay":0.82, "status":"mutation",
     "timestamp":0.75, "tags":[6],  "metrics": None},
    {"id":"7",  "label":"AI & Robotics", "decay":0.65, "status":"predicted",
     "timestamp":0.85, "tags":[7],  "metrics": None},
    {"id":"8",  "label":"Travel",        "decay":0.65, "status":"predicted",
     "timestamp":0.90, "tags":[8],  "metrics": None},
]

_DEMO_LINKS = [
    {"source":"1","target":"2","probability":0.72,"timestamp":0.19},
    {"source":"2","target":"3","probability":0.58,"timestamp":0.37},
    {"source":"1","target":"4","probability":0.45,"timestamp":0.13},
    {"source":"3","target":"6","probability":0.82,"timestamp":0.60},
    {"source":"2","target":"7","probability":0.67,"timestamp":0.57},
    {"source":"3","target":"8","probability":0.53,"timestamp":0.68},
    {"source":"4","target":"5","probability":0.38,"timestamp":0.12},
    {"source":"6","target":"7","probability":0.71,"timestamp":0.80},
]


@router.get("/interest-graph/demo")
def get_demo_graph():
    """Return built-in demo interest graph (always available, no data required)."""
    return {
        "user_id": "demo",
        "nodes":   _DEMO_NODES,
        "links":   _DEMO_LINKS,
    }


# ── Real user graph ───────────────────────────────────────────────────────────

@router.get("/interest-graph/{user_id}")
def get_interest_graph(user_id: int):
    """
    Build and return an interest evolution graph for the given user.

    Returns JSON with:
      nodes: list of InterestNode dicts
      links: list of InterestLink dicts
    """
    # Guard: artifact must be loaded
    if _loader.artifact is None:
        raise HTTPException(status_code=503, detail="Model artifact not loaded yet.")

    # Guard: graph data must be available
    if _loader.interactions_df is None or _loader.tag_matrix is None:
        raise HTTPException(
            status_code=503,
            detail="Interest graph data not ready (interactions_df or tag_matrix missing).",
        )

    # Guard: user must exist in training set
    train_users: list = _loader.artifact.get("train_users", [])
    if user_id not in train_users:
        raise HTTPException(
            status_code=404,
            detail=f"user_id {user_id} not found in training set.",
        )

    nodes, links = build_interest_graph(
        user_id=user_id,
        interactions_df=_loader.interactions_df,
        tag_matrix=_loader.tag_matrix,
        tag_name_fn=_loader.tag_name,
    )

    if not nodes:
        # User exists but has no tag-matched interactions → return demo as fallback
        return {
            "user_id": user_id,
            "fallback": True,
            "nodes":    _DEMO_NODES,
            "links":    _DEMO_LINKS,
        }

    return {
        "user_id": user_id,
        "fallback": False,
        "nodes":    [n.to_dict() for n in nodes],
        "links":    [lk.to_dict() for lk in links],
    }
