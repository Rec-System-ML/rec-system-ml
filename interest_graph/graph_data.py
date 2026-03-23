from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional


VALID_NODE_STATUSES = {"active", "fading", "predicted", "mutation"}


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, float(value)))


@dataclass
class InterestNode:
    """Node model for interest-evolution graph rendering."""

    id: str
    label: str
    decay: float
    status: str
    tags: Optional[list[int]] = None
    timestamp: Optional[float] = None
    metrics: Optional[dict[str, str]] = None

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.label = str(self.label)
        self.decay = _clamp(self.decay)
        if self.status not in VALID_NODE_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}'. Expected one of: {sorted(VALID_NODE_STATUSES)}"
            )
        if self.tags is not None:
            self.tags = [int(tag) for tag in self.tags]
        if self.timestamp is not None:
            self.timestamp = _clamp(self.timestamp)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InterestLink:
    """Edge model for interest-evolution graph rendering."""

    source: str
    target: str
    probability: float
    timestamp: Optional[float] = None

    def __post_init__(self) -> None:
        self.source = str(self.source)
        self.target = str(self.target)
        self.probability = _clamp(self.probability)
        if self.timestamp is not None:
            self.timestamp = _clamp(self.timestamp)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def serialize_graph(
    nodes: Iterable[InterestNode], links: Iterable[InterestLink]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert graph dataclasses into JSON-serializable structures."""

    return [node.to_dict() for node in nodes], [link.to_dict() for link in links]


def demo_graph_data() -> tuple[list[InterestNode], list[InterestLink]]:
    """
    Built-in graph data that matches the concept image style.

    Time coding (decay = brightness, timestamp = when interest emerged):
    - Low decay / low timestamp: dim and dark (past)
    - High decay / high timestamp: bright green and glowing (present)
    - Predicted: purple nodes (near-future)
    - Mutation: red warning node (recent anomaly)

    Timestamp range: 0.0 = 3 months ago, 1.0 = future prediction.
    """

    nodes = [
        InterestNode("history_docs",    "History Documentaries", 0.18, "fading",    tags=[2, 5],       timestamp=0.06),
        InterestNode("travel_blogs",    "Travel Blogs",          0.14, "fading",    tags=[8, 12],      timestamp=0.09),
        InterestNode("classic_lit",     "Classic Literature",    0.44, "fading",    tags=[4, 19],      timestamp=0.22),
        InterestNode("history_stories", "History Stories",       0.24, "fading",    tags=[2, 4, 9],    timestamp=0.16),
        InterestNode("tech_news",       "Tech News",             0.92, "active",    tags=[10, 18, 27], timestamp=0.48, metrics={"ENG": "89%", "CTR": "12%", "WATCH": "42s"}),
        InterestNode("gadget_reviews",  "Gadget Reviews",        0.96, "active",    tags=[10, 29, 34], timestamp=0.62, metrics={"ENG": "94%", "CTR": "18%", "WATCH": "56s"}),
        InterestNode("smart_home",      "Smart Home",            0.94, "active",    tags=[10, 22, 31], timestamp=0.70, metrics={"ENG": "91%", "CTR": "15%", "WATCH": "38s"}),
        InterestNode("baby_products",   "Baby Products",         0.90, "mutation",  tags=[67],         timestamp=0.78),
        InterestNode("ai_research",     "AI Research",           0.72, "predicted", tags=[10, 40],     timestamp=0.90),
        InterestNode("data_science",    "Data Science",          0.69, "predicted", tags=[10, 41],     timestamp=0.94),
    ]

    links = [
        InterestLink("history_docs",   "tech_news",       0.40, timestamp=0.30),
        InterestLink("travel_blogs",   "tech_news",       0.40, timestamp=0.32),
        InterestLink("tech_news",      "gadget_reviews",  0.85, timestamp=0.55),
        InterestLink("tech_news",      "smart_home",      0.85, timestamp=0.58),
        InterestLink("tech_news",      "classic_lit",     0.32, timestamp=0.35),
        InterestLink("gadget_reviews", "smart_home",      0.80, timestamp=0.66),
        InterestLink("smart_home",     "history_stories", 0.46, timestamp=0.42),
        InterestLink("smart_home",     "ai_research",     0.70, timestamp=0.82),
        InterestLink("smart_home",     "data_science",    0.70, timestamp=0.85),
        InterestLink("baby_products",  "tech_news",       0.51, timestamp=0.78),
    ]

    return nodes, links
