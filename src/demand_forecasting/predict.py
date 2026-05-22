from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from demand_forecasting.config import Config
from demand_forecasting.data import prepare_dataset
from demand_forecasting.model import FEATURES
from demand_forecasting.visualize import ensure_dir


def predict_last_period(
    config: Config,
    model_path: str | Path = "models/demand_forecast_model.joblib",
    output_path: str | Path = "outputs/latest_forecast.csv",
) -> pd.DataFrame:
    model = joblib.load(model_path)
    data = prepare_dataset(config.data.sales_path, config.data.product_type)
    latest_week = data["week_start"].max()
    frame = data.loc[data["week_start"].eq(latest_week)].copy()
    frame["prediction"] = model.predict(frame[FEATURES]).clip(min=0)
    result = frame[["week_start", "product_name", "store_location", "prediction"]].sort_values(
        ["store_location", "prediction"], ascending=[True, False]
    )
    output = Path(output_path)
    ensure_dir(output.parent)
    result.to_csv(output, index=False)
    return result
