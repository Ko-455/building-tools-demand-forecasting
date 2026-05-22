from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecasting.config import Config
from demand_forecasting.data import prepare_dataset
from demand_forecasting.model import NUMERIC_FEATURES
from demand_forecasting.train import time_split
from demand_forecasting.visualize import ensure_dir


def population_stability_index(expected, actual, buckets: int = 10) -> float:
    expected = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna()
    if expected.nunique() <= 1 or actual.empty:
        return 0.0

    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(quantiles) < 3:
        return 0.0

    expected_counts = pd.cut(expected, quantiles, include_lowest=True, duplicates="drop").value_counts(sort=False)
    actual_counts = pd.cut(actual, quantiles, include_lowest=True, duplicates="drop").value_counts(sort=False)
    expected_share = (expected_counts / max(expected_counts.sum(), 1)).replace(0, 0.0001)
    actual_share = (actual_counts / max(actual_counts.sum(), 1)).replace(0, 0.0001)
    return float(((actual_share - expected_share) * np.log(actual_share / expected_share)).sum())


def build_monitoring_report(config: Config, reports_dir: str | Path = "reports") -> dict:
    start = time.perf_counter()
    reports_path = ensure_dir(reports_dir)
    data = prepare_dataset(config.data.sales_path, config.data.product_type)
    train, test = time_split(data, config.data.forecast_horizon_weeks)

    missing_share = data.isna().mean().sort_values(ascending=False)
    drift = {
        feature: population_stability_index(train[feature], test[feature])
        for feature in NUMERIC_FEATURES
        if feature in train.columns
    }

    warnings = []
    for col, share in missing_share.items():
        if share > config.monitoring.max_missing_share:
            warnings.append(f"High missing share in {col}: {share:.3f}")
    for feature, psi in drift.items():
        if psi >= config.monitoring.psi_critical_threshold:
            warnings.append(f"Critical drift for {feature}: PSI={psi:.3f}")
        elif psi >= config.monitoring.psi_warning_threshold:
            warnings.append(f"Warning drift for {feature}: PSI={psi:.3f}")

    report = {
        "rows": int(len(data)),
        "series_count": int(data[["product_name", "store_location"]].drop_duplicates().shape[0]),
        "period_min": str(data["week_start"].min().date()),
        "period_max": str(data["week_start"].max().date()),
        "missing_share": {k: float(v) for k, v in missing_share.items()},
        "psi": drift,
        "warnings": warnings,
        "monitoring_seconds": round(time.perf_counter() - start, 3),
    }

    (reports_path / "monitoring_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({"feature": list(drift.keys()), "psi": list(drift.values())}).to_csv(
        reports_path / "drift_report.csv", index=False
    )
    summary = [
        "# Monitoring summary",
        "",
        f"- Rows: {report['rows']}",
        f"- Product-store series: {report['series_count']}",
        f"- Period: {report['period_min']} - {report['period_max']}",
        f"- Warnings: {len(warnings)}",
    ]
    summary.extend(f"- {warning}" for warning in warnings[:10])
    (reports_path / "monitoring_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return report
