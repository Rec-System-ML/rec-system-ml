"""
routers/user.py
---------------
GET /api/users              → available user ID list
GET /api/user/{user_id}/profile → user profile (tag distribution + stats)
"""
from fastapi import APIRouter, HTTPException
import utils.loader as _loader
from utils.loader import get_artifact, get_user_tag_profile

router = APIRouter()


@router.get("/users")
def list_users():
    art = get_artifact()
    users = art.get("train_users", [])
    return {"users": users[:200]}   # cap at 200 for dropdown use


@router.get("/user/{user_id}/profile")
def user_profile(user_id: int):
    art = get_artifact()
    train_users: list = art.get("train_users", [])
    if user_id not in train_users:
        raise HTTPException(status_code=404, detail=f"user_id {user_id} not in training set")

    item_knn = art["item_knn"]
    history = list(item_knn.user_histories.get(user_id, set()))
    click_count = len(history)

    tag_profile = get_user_tag_profile(user_id)

    # Activity percentile: click count ranked against all training users
    all_counts = [len(v) for v in item_knn.user_histories.values()]
    all_counts.sort()
    rank = sum(1 for c in all_counts if c <= click_count)
    activity_pct = round(rank / len(all_counts) * 100) if all_counts else 0

    return {
        "user_id": user_id,
        "click_count": click_count,
        "activity_percentile": activity_pct,
        "register_days": _loader.user_register_days.get(user_id),
        "tag_profile": tag_profile,
        "history_sample": history[:10],
    }
