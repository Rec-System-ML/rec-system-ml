from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

log = logging.getLogger(__name__)

_FALLBACK_NAMES: Dict[int, str] = {
    1: "舞蹈 Dance",
    2: "音乐 Music",
    3: "游戏 Gaming",
    4: "美妆 Beauty",
    5: "穿搭 Fashion",
    6: "娱乐 Entertain",
    7: "运动 Sports",
    8: "颜值 Looks",
    9: "搞笑 Comedy",
    10: "旅游 Travel",
    11: "生活 Lifestyle",
    12: "美食 Food",
    13: "三农 Agriculture",
    14: "教育 Education",
    15: "才艺 Talent",
    16: "健康 Health",
    17: "动物 Pets",
    18: "汽车 Auto",
    19: "情感 Emotion",
    20: "二次元 ACG",
    21: "历史 History",
    22: "财经 Finance",
    23: "宗教 Religion",
    24: "星座命理 Astrology",
    25: "亲子 Parenting",
    26: "摄影 Photo",
    27: "数码科技 DigiTech",
    28: "资讯 News",
    29: "科学 Science",
    30: "法律 Law",
    35: "军事 Military",
    36: "房产家居 Home",
    37: "奇人异象 Bizarre",
    38: "读书 Reading",
}

TAG_DISPLAY_NAMES: Dict[int, str] = {**_FALLBACK_NAMES}

_ID_COL_CANDIDATES = ("tag_id", "category_id", "id", "tag", "tagid")
_NAME_COL_CANDIDATES = (
    "tag_name", "name", "category_name", "category", "label",
    "tag_label", "display_name",
)
_CATEGORIES_FILES: Sequence[str] = (
    "kuairand_video_categories.csv",
    "video_categories.csv",
    "tag_mapping.csv",
    "tag_names.csv",
)


def _pick_col(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def _parse_hierarchical_categories(df: pd.DataFrame) -> Dict[int, str]:
    """Extract unique (category_id → category_name) pairs from a per-video
    hierarchical categories file like ``kuairand_video_categories.csv``.

    Prefers first-level categories; falls back to second-level if needed.
    """
    mapping: Dict[int, str] = {}
    for level in ("first", "second", "third"):
        id_col = f"{level}_level_category_id"
        name_col = f"{level}_level_category_name"
        if id_col not in df.columns or name_col not in df.columns:
            continue
        for _, row in df[[id_col, name_col]].drop_duplicates().iterrows():
            try:
                cid = int(float(row[id_col]))
                cname = str(row[name_col]).strip()
                if cid < 0 or not cname or cname == "UNKNOWN":
                    continue
                if cid not in mapping:
                    mapping[cid] = cname
            except (ValueError, TypeError):
                continue
    return mapping


def load_tag_mapping_csv(data_path: str | Path) -> Dict[int, str]:
    """Try to load a tag-id → display-name mapping CSV from *data_path*.

    Supports two formats:

    1. Simple two-column CSV (``tag_id, tag_name``).
    2. Per-video hierarchical categories (``kuairand_video_categories.csv``)
       with ``first_level_category_id`` / ``first_level_category_name``.

    Returns an empty dict when no suitable file is found.
    """
    base = Path(data_path)
    nested = base / "data" / "KuaiRand-1K" / "data"
    if nested.exists():
        base = nested

    # Also search the directory where this module lives (shared/tag_mapping.csv
    # is committed to the repo so team members always have it after git pull).
    _here = Path(__file__).resolve().parent

    csv_path: Optional[Path] = None
    for name in _CATEGORIES_FILES:
        for search_dir in (base, base.parent, _here):
            candidate = search_dir / name
            if candidate.exists():
                csv_path = candidate
                break
        if csv_path is not None:
            break

    if csv_path is None:
        return {}

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        log.warning("Failed to read tag mapping CSV %s: %s", csv_path, exc)
        return {}

    if "first_level_category_id" in df.columns:
        mapping = _parse_hierarchical_categories(df)
        log.info(
            "Loaded %d category names from hierarchical file %s",
            len(mapping), csv_path,
        )
        return mapping

    id_col = _pick_col(df.columns, _ID_COL_CANDIDATES)
    name_col = _pick_col(df.columns, _NAME_COL_CANDIDATES)

    if id_col is None or name_col is None:
        if len(df.columns) >= 2:
            id_col, name_col = df.columns[0], df.columns[1]
        else:
            log.warning(
                "Cannot auto-detect id/name columns in %s (cols: %s)",
                csv_path, list(df.columns),
            )
            return {}

    mapping: Dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            tid = int(row[id_col])
            tname = str(row[name_col]).strip()
            if tname:
                mapping[tid] = tname
        except (ValueError, TypeError):
            continue

    log.info("Loaded %d tag display names from %s", len(mapping), csv_path)
    return mapping


def update_tag_mapping(mapping: Dict[int, str]) -> int:
    """Merge *mapping* into the global ``TAG_DISPLAY_NAMES`` dict.

    Returns the number of new / updated entries.
    """
    before = len(TAG_DISPLAY_NAMES)
    TAG_DISPLAY_NAMES.update(mapping)
    added = len(TAG_DISPLAY_NAMES) - before
    log.debug("Tag mapping updated: %d new entries, %d total", added, len(TAG_DISPLAY_NAMES))
    return len(mapping)


def ensure_tag_mapping(data_path: str | Path) -> int:
    """Load tag mapping from CSV (if available) and inject into global state.

    Safe to call multiple times; subsequent calls are cheap no-ops when the
    mapping has already been populated beyond the built-in fallback entries.
    """
    if len(TAG_DISPLAY_NAMES) > len(_FALLBACK_NAMES):
        return 0
    mapping = load_tag_mapping_csv(data_path)
    if mapping:
        return update_tag_mapping(mapping)
    return 0


def get_tag_display_name(tag_id: int) -> str:
    return TAG_DISPLAY_NAMES.get(int(tag_id), f"tag_{int(tag_id)}")


def format_tag_ids(tag_ids: Iterable[int], include_raw: bool = True) -> str:
    ids = [int(t) for t in tag_ids]
    names = [get_tag_display_name(t) for t in ids]
    if include_raw:
        raw = ", ".join(f"tag_{t}" for t in ids)
        return f"{', '.join(names)} ({raw})" if ids else ""
    return ", ".join(names)


def format_tags_for_table(tag_ids: Iterable[int]) -> str:
    """DL Table-1 style formatter."""
    return format_tag_ids(tag_ids, include_raw=True)


def format_video_card(video_id: int, tag_ids: Iterable[int], score: float | None = None) -> str:
    """Human-readable card text used in CDS524 demo."""
    tags_text = format_tag_ids(tag_ids, include_raw=False)
    if score is None:
        return f"V_{int(video_id)} | {tags_text}"
    return f"V_{int(video_id)} | {tags_text} | score={float(score):.3f}"
