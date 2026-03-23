"""
api.py  —  KuaiRand-1K Recommendation System — FastAPI entry point
===================================================================
Start:
    uvicorn api:app --reload --port 8100
    # or with custom paths:
    python api.py --artifact checkpoints/mvp_artifact.joblib \
                  --data-dir "../KuaiRand-1K/data"

Dashboard: http://localhost:8100
API docs:  http://localhost:8100/docs
"""
from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path

_back = Path(__file__).resolve().parent
if str(_back) not in sys.path:
    sys.path.insert(0, str(_back))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import utils.loader as loader
from routers import stats, user, recommend, cold_start, interest_graph


DEFAULT_ARTIFACT = _back / "checkpoints" / "mvp_artifact.joblib"
DEFAULT_DATA_DIR = _back.parent / "KuaiRand-1K" / "data"


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args, _ = parser.parse_known_args()
    return Path(args.artifact).expanduser().resolve(), \
           Path(args.data_dir).expanduser().resolve()


_artifact_path, _data_dir = _parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not _artifact_path.exists():
        print(f"[WARNING] artifact not found: {_artifact_path}")
        print("          Run first: python train.py --data-dir <path> --rows 250000")
    elif not _data_dir.exists():
        print(f"[WARNING] data dir not found: {_data_dir}")
    else:
        print(f"[INFO] Loading artifact: {_artifact_path}")
        loader.load_all(_artifact_path, _data_dir)
        print("[INFO] Ready.")
    yield


app = FastAPI(
    title="KuaiRand-1K Recommendation API",
    description="CDS524 backend: ItemKNN + XGBoost CTR + time-decay re-ranking",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stats.router,          prefix="/api", tags=["stats"])
app.include_router(user.router,           prefix="/api", tags=["users"])
app.include_router(recommend.router,      prefix="/api", tags=["recommend"])
app.include_router(cold_start.router,     prefix="/api", tags=["cold-start"])
app.include_router(interest_graph.router, prefix="/api", tags=["interest-graph"])

# Mount ui_prototype/ at root so http://localhost:8100 serves index.html directly
_ui_dir = _back.parent / "ui_prototype"
if _ui_dir.exists():
    app.mount("/", StaticFiles(directory=str(_ui_dir), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8100, reload=True)
