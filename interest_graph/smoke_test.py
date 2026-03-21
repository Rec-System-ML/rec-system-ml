from __future__ import annotations

import argparse
import tempfile
import webbrowser
from pathlib import Path

from . import (
    is_feature_enabled,
    render_demo_interest_graph,
    render_interest_graph,
    set_feature_enabled,
)
from .graph_data import demo_graph_data
from .graph_renderer import TEMPLATE_FILENAME


class SmokeCheckError(RuntimeError):
    pass


def _run_check(step: int, total: int, label: str, condition: bool, detail: str = "") -> None:
    status = "OK" if condition else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"[{step}/{total}] {label:<45} {status}{suffix}")
    if not condition:
        raise SmokeCheckError(f"Check failed: {label}")


def run_smoke_test(open_browser: bool = True) -> Path:
    total = 6
    print("[SMOKE TEST] Interest Evolution Graph")
    print("-" * 54)

    template_path = Path(__file__).with_name(TEMPLATE_FILENAME)
    _run_check(1, total, "Template file exists", template_path.exists(), str(template_path))

    nodes, links = demo_graph_data()
    _run_check(2, total, "Demo data loads", len(nodes) > 0 and len(links) > 0, f"{len(nodes)} nodes / {len(links)} links")

    html = render_demo_interest_graph(width=1024, height=640)
    _run_check(3, total, "Renderer produces HTML (>1KB)", len(html.encode("utf-8")) > 1024, f"{len(html.encode('utf-8'))} bytes")

    _run_check(4, total, "HTML contains D3.js", "d3@" in html or "d3.v" in html)

    labels = [node.label for node in nodes]
    label_hits = sum(1 for label in labels if label in html)
    _run_check(5, total, "HTML contains all node labels", label_hits == len(labels), f"{label_hits}/{len(labels)} found")

    prev_toggle = is_feature_enabled()
    try:
        set_feature_enabled(False)
        fallback_html = render_interest_graph(nodes, links)
    finally:
        set_feature_enabled(prev_toggle)

    fallback_ok = "Fallback" in fallback_html or "plotly" in fallback_html.lower()
    _run_check(6, total, "Fallback renders when disabled", fallback_ok)

    output_path = Path(tempfile.gettempdir()) / "interest_graph_smoke.html"
    output_path.write_text(html, encoding="utf-8")

    print("-" * 54)
    print("ALL CHECKS PASSED.")
    print(f"Output: {output_path}")

    if open_browser:
        opened = webbrowser.open(output_path.as_uri())
        print(f"Browser open: {'YES' if opened else 'NO'}")

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for interest evolution graph module.")
    parser.add_argument("--no-open", action="store_true", help="Do not open browser after generating HTML.")
    args = parser.parse_args()

    try:
        run_smoke_test(open_browser=not args.no_open)
    except SmokeCheckError as exc:
        print(str(exc))
        return 1
    except Exception as exc:  # pragma: no cover - defensive guard for CLI
        print(f"Unexpected error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
