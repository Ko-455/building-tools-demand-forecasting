from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "product_name",
    "brand",
    "product_type",
    "unit",
    "unit_purchase",
    "price_per_unit",
    "total_price",
    "currency",
    "date",
    "delivery",
    "store_location",
    "type_store",
}


def load_sales(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def filter_product_type(df: pd.DataFrame, product_type: str = "Tool") -> pd.DataFrame:
    filtered = df.loc[df["product_type"].eq(product_type)].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for product_type={product_type!r}")
    return filtered


def build_weekly_demand(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["week_start"] = data["date"].dt.to_period("W-MON").dt.start_time
    data["is_delivery"] = data["delivery"].eq("Delivery").astype(int)

    grouped = (
        data.groupby(["week_start", "product_name", "brand", "store_location", "type_store"], as_index=False)
        .agg(
            demand=("unit_purchase", "sum"),
            revenue=("total_price", "sum"),
            avg_price=("price_per_unit", "mean"),
            orders=("unit_purchase", "size"),
            delivery_share=("is_delivery", "mean"),
        )
        .sort_values(["product_name", "store_location", "week_start"])
    )

    calendar = pd.date_range(grouped["week_start"].min(), grouped["week_start"].max(), freq="W-TUE")
    keys = grouped[["product_name", "brand", "store_location", "type_store"]].drop_duplicates()
    full_index = keys.merge(pd.DataFrame({"week_start": calendar}), how="cross")
    weekly = full_index.merge(
        grouped,
        on=["week_start", "product_name", "brand", "store_location", "type_store"],
        how="left",
    )
    fill_zero = ["demand", "revenue", "orders", "delivery_share"]
    weekly[fill_zero] = weekly[fill_zero].fillna(0)
    weekly["avg_price"] = weekly.groupby("product_name")["avg_price"].transform(lambda s: s.ffill().bfill())
    weekly["avg_price"] = weekly["avg_price"].fillna(weekly["avg_price"].median())
    return weekly.sort_values(["product_name", "store_location", "week_start"]).reset_index(drop=True)


def add_time_series_features(weekly: pd.DataFrame) -> pd.DataFrame:
    data = weekly.copy()
    group_cols = ["product_name", "store_location"]
    data["week"] = data["week_start"].dt.isocalendar().week.astype(int)
    data["month"] = data["week_start"].dt.month
    data["quarter"] = data["week_start"].dt.quarter
    data["year"] = data["week_start"].dt.year
    data["week_index"] = ((data["week_start"] - data["week_start"].min()).dt.days // 7).astype(int)

    for lag in (1, 2, 4, 8):
        data[f"lag_{lag}"] = data.groupby(group_cols)["demand"].shift(lag)

    shifted = data.groupby(group_cols)["demand"].shift(1)
    data["rolling_mean_4"] = shifted.groupby([data["product_name"], data["store_location"]]).rolling(4, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    data["rolling_mean_8"] = shifted.groupby([data["product_name"], data["store_location"]]).rolling(8, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    data["rolling_std_4"] = shifted.groupby([data["product_name"], data["store_location"]]).rolling(4, min_periods=1).std().reset_index(level=[0, 1], drop=True)

    feature_cols = [c for c in data.columns if c.startswith("lag_") or c.startswith("rolling_")]
    data[feature_cols] = data[feature_cols].fillna(0)
    data["rolling_std_4"] = data["rolling_std_4"].replace([np.inf, -np.inf], 0).fillna(0)
    return data


def prepare_dataset(path: str | Path, product_type: str = "Tool") -> pd.DataFrame:
    raw = load_sales(path)
    tools = filter_product_type(raw, product_type)
    weekly = build_weekly_demand(tools)
    return add_time_series_features(weekly)
