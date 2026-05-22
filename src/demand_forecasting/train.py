from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd

from demand_forecasting.config import Config
from demand_forecasting.data import prepare_dataset
from demand_forecasting.metrics import regression_metrics
from demand_forecasting.model import FEATURES, TARGET, build_automl_search
from demand_forecasting.visualize import (
    ensure_dir,
    plot_forecast_vs_actual,
    plot_residuals,
    plot_top_products,
    plot_weekly_demand,
)


def time_split(data: pd.DataFrame, horizon_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = data["week_start"].max() - pd.Timedelta(weeks=horizon_weeks - 1)
    train = data.loc[data["week_start"] < cutoff].copy()
    test = data.loc[data["week_start"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Time split produced an empty train or test set")
    return train, test


def train_pipeline(
    config: Config,
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
) -> dict:
    start = time.perf_counter()
    models_path = ensure_dir(models_dir)
    reports_path = ensure_dir(reports_dir)
    figures_path = ensure_dir(Path(reports_dir) / "figures")

    data = prepare_dataset(config.data.sales_path, config.data.product_type)
    train, test = time_split(data, config.data.forecast_horizon_weeks)

    search = build_automl_search(
        random_state=config.training.random_state,
        cv_splits=config.training.cv_splits,
        scoring=config.training.scoring,
        n_jobs=config.training.n_jobs,
    )
    search.fit(train[FEATURES], train[TARGET])
    predictions = search.predict(test[FEATURES])
    metrics = regression_metrics(test[TARGET], predictions)

    model_path = models_path / "demand_forecast_model.joblib"
    joblib.dump(search.best_estimator_, model_path)

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results.to_csv(reports_path / "automl_cv_results.csv", index=False)

    prediction_frame = test[["week_start", "product_name", "store_location", "demand"]].copy()
    prediction_frame["prediction"] = predictions.clip(min=0)
    prediction_frame.to_csv(reports_path / "holdout_predictions.csv", index=False)

    artifacts = {
        "weekly_demand": str(plot_weekly_demand(data, figures_path)),
        "forecast_vs_actual": str(plot_forecast_vs_actual(test, predictions.clip(min=0), figures_path)),
        "top_products": str(plot_top_products(data, figures_path)),
        "residuals": str(plot_residuals(test, predictions.clip(min=0), figures_path)),
    }

    summary = {
        "model_path": str(model_path),
        "rows_total": int(len(data)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "date_min": str(data["week_start"].min().date()),
        "date_max": str(data["week_start"].max().date()),
        "holdout_weeks": int(config.data.forecast_horizon_weeks),
        "best_estimator": type(search.best_estimator_.named_steps["model"]).__name__,
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "best_cv_score": float(search.best_score_),
        "metrics": metrics,
        "training_seconds": round(time.perf_counter() - start, 3),
        "artifacts": artifacts,
    }
    (reports_path / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
