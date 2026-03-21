from __future__ import annotations

import os
from typing import Any, Iterable

from .graph_data import InterestLink, InterestNode, demo_graph_data, serialize_graph


def _read_toggle_from_env() -> bool:
    return os.environ.get("INTEREST_GRAPH_ENABLED", "1").strip() == "1"


FEATURE_ENABLED = _read_toggle_from_env()


def set_feature_enabled(enabled: bool) -> None:
    """Runtime toggle for enabling/disabling the advanced interest graph."""
    global FEATURE_ENABLED
    FEATURE_ENABLED = bool(enabled)


def is_feature_enabled() -> bool:
    """Return current in-memory feature-toggle state."""
    return FEATURE_ENABLED


def render_interest_graph(
    nodes: Iterable[InterestNode | dict[str, Any]],
    links: Iterable[InterestLink | dict[str, Any]],
    **kwargs: Any,
) -> str:
    """
    Return embeddable HTML for interest graph.

    Toggle behavior:
    - FEATURE_ENABLED=True: advanced D3/SVG/Canvas renderer
    - FEATURE_ENABLED=False: degraded fallback renderer
    """
    if not FEATURE_ENABLED:
        from .fallback import render_fallback

        return render_fallback(nodes=nodes, links=links)

    from .graph_renderer import render

    return render(nodes=nodes, links=links, **kwargs)


def render_demo_interest_graph(**kwargs: Any) -> str:
    """Render built-in demo graph with toggle-aware behavior."""
    demo_nodes, demo_links = demo_graph_data()
    return render_interest_graph(nodes=demo_nodes, links=demo_links, **kwargs)


def streamlit_interest_graph(
    nodes: Iterable[InterestNode | dict[str, Any]],
    links: Iterable[InterestLink | dict[str, Any]],
    height: int = 650,
    scrolling: bool = False,
    **kwargs: Any,
) -> None:
    """Streamlit helper that renders the graph HTML into the app."""
    try:
        import streamlit.components.v1 as components
    except Exception as exc:
        raise RuntimeError(
            "streamlit is required for streamlit_interest_graph()."
        ) from exc

    html = render_interest_graph(nodes=nodes, links=links, **kwargs)
    components.html(html, height=height, scrolling=scrolling)


__all__ = [
    "FEATURE_ENABLED",
    "InterestLink",
    "InterestNode",
    "demo_graph_data",
    "is_feature_enabled",
    "render_demo_interest_graph",
    "render_interest_graph",
    "serialize_graph",
    "set_feature_enabled",
    "streamlit_interest_graph",
]
