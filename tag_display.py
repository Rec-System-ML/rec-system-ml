from __future__ import annotations

from typing import Iterable, List


# Placeholder display names. Update this map after inspecting real tag distribution.
TAG_DISPLAY_NAMES = {
    1: "Lifestyle",
    2: "Entertainment",
    3: "Food",
    4: "Music",
    5: "Sports",
    6: "Tech",
    7: "Education",
    8: "Travel",
}


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
    """CDS525 Table-1 style formatter."""
    return format_tag_ids(tag_ids, include_raw=True)


def format_video_card(video_id: int, tag_ids: Iterable[int], score: float | None = None) -> str:
    """Human-readable card text used in CDS524 demo."""
    tags_text = format_tag_ids(tag_ids, include_raw=False)
    if score is None:
        return f"V_{int(video_id)} | {tags_text}"
    return f"V_{int(video_id)} | {tags_text} | score={float(score):.3f}"
