from demand_forecasting.model import FEATURES, TARGET, build_preprocessor


def test_feature_list_contains_target_separately():
    assert TARGET == "demand"
    assert TARGET not in FEATURES
    assert "product_name" in FEATURES
    assert "lag_1" in FEATURES


def test_preprocessor_has_expected_transformers():
    preprocessor = build_preprocessor()
    names = [name for name, _, _ in preprocessor.transformers]
    assert names == ["cat", "num"]
