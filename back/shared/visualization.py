from __future__ import annotations

from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_training_curves(
    history: pd.DataFrame,
    epoch_col: str = "epoch",
    train_col: str = "train_loss",
    val_col: str = "val_loss",
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history[epoch_col], history[train_col], label=train_col)
    if val_col in history.columns:
        ax.plot(history[epoch_col], history[val_col], label=val_col)
    ax.set_title("Training Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_metric_bars(metrics: Dict[str, float], title: str = "Metric Comparison"):
    df = pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="metric", y="value", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=20)
    return fig


def plot_model_comparison(
    model_metrics: pd.DataFrame,
    model_col: str = "model",
    metric_col: str = "metric",
    value_col: str = "value",
):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=model_metrics, x=metric_col, y=value_col, hue=model_col, ax=ax)
    ax.set_title("Model Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.legend(title="Model")
    return fig


def plot_dl_training_panel(
    history: pd.DataFrame,
    epoch_col: str = "epoch",
    train_loss_col: str = "train_loss",
    train_acc_col: str = "train_acc",
    test_acc_col: str = "test_acc",
    title: str = "Training Loss/Accuracy vs Epoch",
):
    """
    DL-style figure with one loss curve and two accuracy curves.
    Left y-axis: loss, right y-axis: accuracies.
    """
    fig, ax_loss = plt.subplots(figsize=(9, 5))
    ax_acc = ax_loss.twinx()

    ax_loss.plot(history[epoch_col], history[train_loss_col], color="tab:blue", label="Training Loss")
    ax_acc.plot(history[epoch_col], history[train_acc_col], color="tab:green", linestyle="--", label="Training Accuracy")
    if test_acc_col in history.columns:
        ax_acc.plot(history[epoch_col], history[test_acc_col], color="tab:red", linestyle="--", label="Test Accuracy")

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_acc.set_ylabel("Accuracy")
    ax_loss.set_title(title)
    ax_loss.grid(alpha=0.3)

    lines = ax_loss.get_lines() + ax_acc.get_lines()
    labels = [line.get_label() for line in lines]
    ax_loss.legend(lines, labels, loc="best")
    return fig
