"""
api.py  —  KuaiRand-1K 推荐系统 FastAPI 主入口
================================================
启动方式:
    uvicorn api:app --reload --port 8100
    # 或指定 artifact / data 路径:
    python api.py --artifact checkpoints/mvp_artifact.joblib \
                  --data-dir "../KuaiRand-1K/data"

前端访问: http://localhost:8100
API 文档: http://localhost:8100/docs
"""
from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# 确保 back/ 目录可导入（models/, routers/, utils/）
_back = Path(__file__).resolve().parent
if str(_back) not in sys.path:
    sys.path.insert(0, str(_back))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import utils.loader as loader
from routers import stats, user, recommend, cold_start


# ── 默认路径 ─────────────────────────────────────────────────────────────────

DEFAULT_ARTIFACT = _back / "checkpoints" / "mvp_artifact.joblib"
DEFAULT_DATA_DIR = _back.parent / "KuaiRand-1K" / "data"


# ── 解析命令行（允许 uvicorn 直接启动时不传参） ───────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args, _ = parser.parse_known_args()
    return Path(args.artifact).expanduser().resolve(), \
           Path(args.data_dir).expanduser().resolve()


_artifact_path, _data_dir = _parse_args()


# ── Lifespan：启动时加载数据 ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not _artifact_path.exists():
        print(f"[WARNING] artifact not found: {_artifact_path}")
        print("          先运行: python train.py --data-dir <path> --rows 250000")
    elif not _data_dir.exists():
        print(f"[WARNING] data dir not found: {_data_dir}")
    else:
        print(f"[INFO] Loading artifact: {_artifact_path}")
        loader.load_all(_artifact_path, _data_dir)
        print("[INFO] Ready.")
    yield


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="KuaiRand-1K 推荐系统 API",
    description="CDS524 项目后端：ItemKNN + XGBoost CTR + 时间衰减重排",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS：允许本地前端 HTML 文件直接调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由注册
app.include_router(stats.router,       prefix="/api", tags=["统计"])
app.include_router(user.router,        prefix="/api", tags=["用户"])
app.include_router(recommend.router,   prefix="/api", tags=["推荐"])
app.include_router(cold_start.router,  prefix="/api", tags=["冷启动"])

# 静态文件：把 ui_prototype/ 挂载到根路径，访问 / 直接显示 index.html
_ui_dir = _back.parent / "ui_prototype"
if _ui_dir.exists():
    app.mount("/", StaticFiles(directory=str(_ui_dir), html=True), name="ui")


# ── 直接 python api.py 启动 ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8100, reload=True)
