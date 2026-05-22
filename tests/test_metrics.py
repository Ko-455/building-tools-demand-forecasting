from demand_forecasting.metrics import regression_metrics, smape


def test_smape_is_zero_for_equal_values():
    assert smape([1, 2, 3], [1, 2, 3]) == 0


def test_regression_metrics_returns_expected_keys():
    metrics = regression_metrics([10, 20, 30], [12, 18, 33])
    assert {"mae", "rmse", "mape_nonzero", "smape", "r2"} <= set(metrics)
    assert metrics["mae"] > 0
