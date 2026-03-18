from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("mlflow is not installed. Run: pip install mlflow") from exc
    return mlflow


def start_experiment(
    experiment_name: str,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Any:
    """Start (or resume) an MLflow run."""
    mlflow = _mlflow()
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name, tags=tags)


def end_experiment(status: str = "FINISHED") -> None:
    """End the active MLflow run."""
    mlflow = _mlflow()
    mlflow.end_run(status=status)


def _log_payload(payload: Dict[str, Any], step: Optional[int] = None) -> None:
    """
    Log a mixed payload.
    - numeric values -> metrics
    - string values -> params
    """
    metrics: Dict[str, float] = {}
    params: Dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
        else:
            params[key] = str(value)
    if params:
        log_params(params)
    if metrics:
        log_metrics(metrics, step=step)


def log_experiment(*args: Any, **kwargs: Any) -> None:
    """
    Compatibility wrapper for two calling styles.

    Style A (epoch logging):
        log_experiment({"train_loss": 0.12, "run_tag": "baseline"}, step=3)

    Style B (single-run summary):
        log_experiment(
            model_name="XGBoost_CTR",
            params={"n_estimators": 100},
            metrics={"AUC": 0.71},
            artifacts=["figures/shap_summary.png"],
            experiment_name="recsys"
        )
    """
    if args and isinstance(args[0], dict):
        payload = args[0]
        step = kwargs.get("step")
        _log_payload(payload, step=step)
        return

    model_name = kwargs.get("model_name")
    params = kwargs.get("params", {}) or {}
    metrics = kwargs.get("metrics", {}) or {}
    artifacts = kwargs.get("artifacts")
    experiment_name = kwargs.get("experiment_name", "recsys")

    if model_name is None:
        raise TypeError("log_experiment requires either payload dict or model_name=...")

    mlflow = _mlflow()
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=str(model_name)):
        if params:
            log_params(params)
        if metrics:
            log_metrics(metrics)
        if artifacts:
            log_artifacts(artifacts)


def log_params(params: Dict[str, Any]) -> None:
    mlflow = _mlflow()
    clean = {k: str(v) for k, v in params.items()}
    mlflow.log_params(clean)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    mlflow = _mlflow()
    if step is None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
    else:
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value), step=step)


def log_artifacts(paths: Iterable[str | Path], artifact_path: Optional[str] = None) -> None:
    mlflow = _mlflow()
    for p in paths:
        path = Path(p)
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_figure(fig: Any, artifact_file: str) -> None:
    mlflow = _mlflow()
    mlflow.log_figure(fig, artifact_file=artifact_file)


def log_shap_plots(figs: Dict[str, Any], task_name: Optional[str] = None, close_figures: bool = True) -> None:
    """Log SHAP matplotlib figures to MLflow with stable artifact names."""
    suffix = f"_{task_name}" if task_name else ""
    for name, fig in figs.items():
        artifact_file = f"shap/{name}{suffix}.png"
        log_figure(fig, artifact_file=artifact_file)
        if close_figures:
            import matplotlib.pyplot as plt

            plt.close(fig)


def log_model(model: Any, artifact_path: str = "model", flavor: str = "pickle") -> None:
    """
    Log model artifact with a lightweight flavor switch.
    flavor: sklearn | pytorch | pickle
    """
    mlflow = _mlflow()
    flavor = flavor.lower()
    if flavor == "sklearn":
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
    elif flavor == "pytorch":
        mlflow.pytorch.log_model(model, artifact_path=artifact_path)
    else:
        import pickle
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.pkl"
            with model_path.open("wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(str(model_path), artifact_path=artifact_path)
