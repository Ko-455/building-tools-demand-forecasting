import pandas as pd

from demand_forecasting.data import add_time_series_features, build_weekly_demand, filter_product_type


def sample_sales():
    return pd.DataFrame(
        {
            "product_name": ["Hammer", "Hammer", "Cement Bag"],
            "brand": ["Lippro", "Lippro", "Semen Tiga Roda"],
            "product_type": ["Tool", "Tool", "Cement"],
            "unit": ["Piece", "Piece", "Sack"],
            "unit_purchase": [2, 3, 5],
            "price_per_unit": [50000, 50000, 70000],
            "total_price": [100000, 150000, 350000],
            "currency": ["IDR", "IDR", "IDR"],
            "date": pd.to_datetime(["2024-01-03", "2024-01-05", "2024-01-06"]),
            "delivery": ["Delivery", "Pickup", "Pickup"],
            "store_location": ["Jakarta", "Jakarta", "Jakarta"],
            "type_store": ["Branch", "Branch", "Branch"],
        }
    )


def test_filter_product_type_keeps_only_tools():
    filtered = filter_product_type(sample_sales(), "Tool")
    assert filtered["product_type"].eq("Tool").all()
    assert len(filtered) == 2


def test_weekly_demand_aggregates_units():
    weekly = build_weekly_demand(filter_product_type(sample_sales(), "Tool"))
    assert weekly["demand"].sum() == 5
    assert weekly.loc[weekly["demand"].gt(0), "orders"].iloc[0] == 2


def test_time_series_features_have_lags():
    weekly = build_weekly_demand(filter_product_type(sample_sales(), "Tool"))
    featured = add_time_series_features(weekly)
    for column in ["week", "month", "lag_1", "rolling_mean_4"]:
        assert column in featured.columns
    assert featured[["lag_1", "rolling_mean_4"]].isna().sum().sum() == 0
