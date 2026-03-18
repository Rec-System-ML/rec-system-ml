from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _to_numpy_2d(x: Any) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _flatten_feature_dict(feature_dict: Dict[str, Any]) -> np.ndarray:
    parts = []
    for _, value in feature_dict.items():
        arr = _to_numpy_2d(value)
        parts.append(arr)
    return np.concatenate(parts, axis=1) if parts else np.empty((0, 0))


def _collect_samples(test_data: Any, max_rows: int = 2_000) -> np.ndarray:
    """
    Convert common test_data containers to a 2D numpy array.

    Supports:
    - pandas/numpy like arrays
    - iterable DataLoader-like batches with keys: "x"/"features"/("user","video")
    """
    if hasattr(test_data, "to_numpy") or isinstance(test_data, np.ndarray):
        return _to_numpy_2d(test_data)

    rows = []
    if hasattr(test_data, "__iter__"):
        for batch in test_data:
            if isinstance(batch, dict):
                if "x" in batch:
                    rows.append(_to_numpy_2d(batch["x"]))
                elif "features" in batch:
                    rows.append(_to_numpy_2d(batch["features"]))
                elif "user" in batch and "video" in batch:
                    user = _flatten_feature_dict(batch["user"]) if isinstance(batch["user"], dict) else _to_numpy_2d(batch["user"])
                    video = _flatten_feature_dict(batch["video"]) if isinstance(batch["video"], dict) else _to_numpy_2d(batch["video"])
                    rows.append(np.concatenate([user, video], axis=1))
                else:
                    continue
            else:
                rows.append(_to_numpy_2d(batch))
            if rows and sum(len(r) for r in rows) >= max_rows:
                break
    if not rows:
        raise TypeError("Unsupported test_data type for SHAP analysis.")
    x = np.vstack(rows)
    return x[:max_rows]


def _build_predict_fn(model: Any, task_id: Optional[int] = None) -> Callable[[np.ndarray], np.ndarray]:
    if hasattr(model, "predict_proba"):
        return lambda x: model.predict_proba(x)[:, 1]
    if hasattr(model, "predict"):
        return lambda x: np.asarray(model.predict(x)).reshape(-1)
    if callable(model):
        def _fn(x: np.ndarray) -> np.ndarray:
            out = model(x)
            # Handle pytorch tensor outputs without requiring torch dependency at import time.
            if hasattr(out, "detach") and hasattr(out, "cpu"):
                out = out.detach().cpu().numpy()
            if isinstance(out, (list, tuple)) and task_id is not None:
                selected = out[task_id]
                if hasattr(selected, "detach") and hasattr(selected, "cpu"):
                    selected = selected.detach().cpu().numpy()
                return np.asarray(selected).reshape(-1)
            out_arr = np.asarray(out)
            if out_arr.ndim == 2 and task_id is not None:
                return out_arr[:, task_id]
            return out_arr.reshape(-1)
        return _fn
    raise TypeError("model must expose predict_proba/predict or be callable.")


def run_shap_analysis(
    model: Any,
    test_data: Any,
    feature_names: Optional[Iterable[str]] = None,
    task_id: Optional[int] = None,
    model_type: str = "auto",
    task_name: Optional[str] = None,
    background_size: int = 100,
    sample_size: int = 200,
    random_state: int = 42,
    save_dir: Optional[str | Path] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute SHAP values and return a small figure bundle.
    """
    import shap

    x = _collect_samples(test_data)
    rng = np.random.default_rng(random_state)
    bg_idx = rng.choice(len(x), size=min(background_size, len(x)), replace=False)
    sample_idx = rng.choice(len(x), size=min(sample_size, len(x)), replace=False)
    background = x[bg_idx]
    samples = x[sample_idx]

    predict_fn = _build_predict_fn(model=model, task_id=task_id)
    use_tree = model_type == "tree" or (model_type == "auto" and hasattr(model, "predict_proba") and hasattr(model, "feature_importances_"))
    if use_tree:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(samples)
    else:
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(samples)
    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim > 2:
        shap_arr = shap_arr[0]

    figs: Dict[str, Any] = {}

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_arr, samples, feature_names=feature_names, show=False, plot_type="dot")
    figs["summary_dot"] = plt.gcf()

    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_arr, samples, feature_names=feature_names, show=False, plot_type="bar")
    figs["summary_bar"] = plt.gcf()

    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        suffix = f"_{task_name}" if task_name else ""
        figs["summary_dot"].savefig(out / f"shap_summary_dot{suffix}.png", dpi=300, bbox_inches="tight")
        figs["summary_bar"].savefig(out / f"shap_summary_bar{suffix}.png", dpi=300, bbox_inches="tight")

    return shap_arr, figs


def generate_text_explanation(
    shap_values: np.ndarray,
    feature_names: Iterable[str],
    sample_idx: int = 0,
    top_k: int = 3,
) -> str:
    """Convert one sample's SHAP values into short natural-language explanation."""
    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim == 1:
        sample_shap = shap_arr
    else:
        sample_idx = max(0, min(sample_idx, shap_arr.shape[0] - 1))
        sample_shap = shap_arr[sample_idx]

    names = list(feature_names)
    if len(names) != len(sample_shap):
        names = [f"feature_{i}" for i in range(len(sample_shap))]

    top_indices = np.argsort(np.abs(sample_shap))[-top_k:][::-1]
    pieces = []
    for idx in top_indices:
        direction = "increase" if sample_shap[idx] >= 0 else "decrease"
        pieces.append(f"{names[idx]} ({direction} {abs(float(sample_shap[idx])):.4f})")
    return "Top drivers: " + ", ".join(pieces)
