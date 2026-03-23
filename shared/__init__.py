"""Shared toolkit for CDS524/CDS525 recommendation projects."""

from .data_pipeline import load_data, preprocess_kuairand
from .evaluation import (
    calculate_coverage,
    calculate_gini_coefficient,
    compute_classification_metrics,
    evaluate_model,
    evaluate_ranking,
)
from .reranker import TimeDecayReranker
from .shap_utils import generate_text_explanation, run_shap_analysis
from .tag_display import (
    ensure_tag_mapping,
    format_tag_ids,
    format_tags_for_table,
    format_video_card,
    get_tag_display_name,
    load_tag_mapping_csv,
    update_tag_mapping,
)

__all__ = [
    "TimeDecayReranker",
    "calculate_coverage",
    "calculate_gini_coefficient",
    "compute_classification_metrics",
    "evaluate_model",
    "evaluate_ranking",
    "ensure_tag_mapping",
    "format_tag_ids",
    "format_tags_for_table",
    "format_video_card",
    "generate_text_explanation",
    "get_tag_display_name",
    "load_data",
    "load_tag_mapping_csv",
    "preprocess_kuairand",
    "run_shap_analysis",
    "update_tag_mapping",
]

