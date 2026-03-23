from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from .graph_data import InterestLink, InterestNode, serialize_graph


TEMPLATE_FILENAME = "template.html"


def _load_template() -> str:
    template_path = Path(__file__).with_name(TEMPLATE_FILENAME)
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _to_dict_list(items: Iterable[Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, (InterestNode, InterestLink)):
            output.append(item.to_dict())
        elif is_dataclass(item):
            output.append(asdict(item))
        elif isinstance(item, dict):
            output.append(item)
        else:
            raise TypeError(
                "items must be InterestNode/InterestLink dataclass objects or dicts"
            )
    return output


def render(
    nodes: Iterable[InterestNode | dict[str, Any]],
    links: Iterable[InterestLink | dict[str, Any]],
    width: int = 1024,
    height: int = 640,
    show_timeline: bool = True,
    show_info_panels: bool = True,
) -> str:
    """
    Render the sci-fi interest-evolution graph into embeddable HTML.

    Returns a full HTML document string that can be embedded in Streamlit via:
    streamlit.components.v1.html(...)
    """

    if width < 640:
        raise ValueError("width must be >= 640")
    if height < 420:
        raise ValueError("height must be >= 420")

    template = _load_template()

    node_dicts = _to_dict_list(nodes)
    link_dicts = _to_dict_list(links)

    html = (
        template.replace("__NODES_DATA__", json.dumps(node_dicts, ensure_ascii=False))
        .replace("__LINKS_DATA__", json.dumps(link_dicts, ensure_ascii=False))
        .replace("__WIDTH__", str(int(width)))
        .replace("__HEIGHT__", str(int(height)))
        .replace("__SHOW_TIMELINE__", json.dumps(bool(show_timeline)))
        .replace("__SHOW_INFO__", json.dumps(bool(show_info_panels)))
    )

    return html


def render_demo(
    width: int = 1024,
    height: int = 640,
    show_timeline: bool = True,
    show_info_panels: bool = True,
) -> str:
    """Render the built-in demo graph as HTML."""
    from .graph_data import demo_graph_data

    nodes, links = demo_graph_data()
    return render(
        nodes=nodes,
        links=links,
        width=width,
        height=height,
        show_timeline=show_timeline,
        show_info_panels=show_info_panels,
    )


def write_html(
    output_path: str | Path,
    nodes: Iterable[InterestNode | dict[str, Any]],
    links: Iterable[InterestLink | dict[str, Any]],
    **render_kwargs: Any,
) -> Path:
    """Render and write HTML to disk for standalone preview."""

    html = render(nodes=nodes, links=links, **render_kwargs)
    path = Path(output_path).expanduser().resolve()
    path.write_text(html, encoding="utf-8")
    return path
