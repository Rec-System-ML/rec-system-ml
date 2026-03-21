from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable


def _to_dict_list(items: Iterable[Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            output.append(item)
        elif is_dataclass(item):
            output.append(asdict(item))
        else:
            raise TypeError("Fallback renderer expects dict or dataclass items")
    return output


def _circle_layout(node_count: int, radius: float = 0.95) -> list[tuple[float, float]]:
    if node_count == 0:
        return []
    if node_count == 1:
        return [(0.0, 0.0)]

    positions: list[tuple[float, float]] = []
    for idx in range(node_count):
        theta = (2 * math.pi * idx) / node_count
        positions.append((math.cos(theta) * radius, math.sin(theta) * radius))
    return positions


def _color_for_status(status: str) -> str:
    if status == "mutation":
        return "#ff5959"
    if status == "predicted":
        return "#c4a2ff"
    if status == "active":
        return "#24f7a2"
    return "#8ea0bf"


def _render_plain_html(nodes: list[dict[str, Any]], links: list[dict[str, Any]]) -> str:
    payload = {
        "nodes": [{"label": n.get("label"), "status": n.get("status"), "decay": n.get("decay")} for n in nodes],
        "links": [{"source": l.get("source"), "target": l.get("target"), "probability": l.get("probability")} for l in links],
    }
    pretty = json.dumps(payload, indent=2)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Interest Graph Fallback</title>
  <style>
    body {{
      margin: 0;
      padding: 20px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      background: #0f172a;
      color: #d8e9ff;
    }}
    .card {{
      border: 1px solid #334155;
      border-radius: 10px;
      padding: 14px;
      background: #111b2f;
    }}
    h3 {{ margin-top: 0; color: #7dd3fc; }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="card">
    <h3>Interest Graph Fallback (plotly unavailable)</h3>
    <pre>{pretty}</pre>
  </div>
</body>
</html>
"""


def render_fallback(
    nodes: Iterable[Any],
    links: Iterable[Any],
) -> str:
    """
    Degraded fallback renderer for the interest graph.

    Preferred output: Plotly dark network chart.
    Hard fallback: plain HTML payload when Plotly is unavailable.
    """

    node_dicts = _to_dict_list(nodes)
    link_dicts = _to_dict_list(links)

    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception:
        return _render_plain_html(node_dicts, link_dicts)

    index_by_id = {str(node["id"]): idx for idx, node in enumerate(node_dicts)}
    positions = _circle_layout(len(node_dicts), radius=1.0)

    edge_x: list[float] = []
    edge_y: list[float] = []
    annotations = []
    for link in link_dicts:
        source_idx = index_by_id.get(str(link["source"]))
        target_idx = index_by_id.get(str(link["target"]))
        if source_idx is None or target_idx is None:
            continue

        sx, sy = positions[source_idx]
        tx, ty = positions[target_idx]
        edge_x.extend([sx, tx, None])
        edge_y.extend([sy, ty, None])
        probability = float(link.get("probability", 0.0))
        annotations.append(
            dict(
                x=(sx + tx) / 2,
                y=(sy + ty) / 2,
                text=f"{probability:.2f}",
                showarrow=False,
                font=dict(size=10, color="#cde6ff"),
            )
        )

    node_x = [pos[0] for pos in positions]
    node_y = [pos[1] for pos in positions]
    node_text = [str(node.get("label", node.get("id", "node"))) for node in node_dicts]
    node_sizes = [20 + max(0.0, min(1.0, float(node.get("decay", 0.0)))) * 18 for node in node_dicts]
    node_colors = [_color_for_status(str(node.get("status", "fading"))) for node in node_dicts]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.2, color="rgba(134, 181, 255, 0.5)"),
        hoverinfo="none",
        mode="lines",
    )
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="middle center",
        hovertemplate="Label: %{text}<extra></extra>",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1.5, color="rgba(220, 238, 255, 0.5)"),
            opacity=0.95,
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Interest Evolution Graph (Fallback Mode)",
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="#0a0e17",
        paper_bgcolor="#0a0e17",
        annotations=annotations,
    )
    return pio.to_html(fig, full_html=True, include_plotlyjs="cdn")
