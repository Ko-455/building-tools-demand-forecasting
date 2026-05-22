from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_weekly_demand(data: pd.DataFrame, output_dir: str | Path) -> Path:
    out_dir = ensure_dir(output_dir)
    weekly = data.groupby("week_start", as_index=False)["demand"].sum()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=weekly, x="week_start", y="demand", ax=ax, color="#2563eb")
    ax.set_title("Weekly demand for hand building tools")
    ax.set_xlabel("Week")
    ax.set_ylabel("Units")
    fig.tight_layout()
    path = out_dir / "weekly_tool_demand.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_forecast_vs_actual(test_data: pd.DataFrame, predictions, output_dir: str | Path) -> Path:
    out_dir = ensure_dir(output_dir)
    plot_data = test_data[["week_start", "demand"]].copy()
    plot_data["prediction"] = predictions
    plot_data = plot_data.groupby("week_start", as_index=False)[["demand", "prediction"]].sum()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(plot_data["week_start"], plot_data["demand"], label="Actual", color="#111827", linewidth=2)
    ax.plot(plot_data["week_start"], plot_data["prediction"], label="Forecast", color="#dc2626", linewidth=2)
    ax.set_title("Forecast vs actual demand on holdout period")
    ax.set_xlabel("Week")
    ax.set_ylabel("Units")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "forecast_vs_actual.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_top_products(data: pd.DataFrame, output_dir: str | Path) -> Path:
    out_dir = ensure_dir(output_dir)
    top = (
        data.groupby("product_name", as_index=False)["demand"]
        .sum()
        .sort_values("demand", ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top, y="product_name", x="demand", ax=ax, color="#0f766e")
    ax.set_title("Top 10 tool products by demand")
    ax.set_xlabel("Units")
    ax.set_ylabel("")
    fig.tight_layout()
    path = out_dir / "top_tool_products.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_residuals(test_data: pd.DataFrame, predictions, output_dir: str | Path) -> Path:
    out_dir = ensure_dir(output_dir)
    residuals = test_data["demand"].to_numpy() - predictions
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(residuals, bins=30, kde=True, ax=ax, color="#7c3aed")
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title("Forecast residual distribution")
    ax.set_xlabel("Actual - forecast")
    fig.tight_layout()
    path = out_dir / "residuals.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
