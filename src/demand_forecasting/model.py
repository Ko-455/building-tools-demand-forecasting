from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = ["product_name", "brand", "store_location", "type_store"]
NUMERIC_FEATURES = [
    "avg_price",
    "orders",
    "delivery_share",
    "week",
    "month",
    "quarter",
    "year",
    "week_index",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_std_4",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "demand"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_automl_search(random_state: int, cv_splits: int, scoring: str, n_jobs: int) -> GridSearchCV:
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", Ridge()),
        ]
    )

    param_grid = [
        {
            "model": [Ridge(random_state=random_state)],
            "model__alpha": [0.1, 1.0, 10.0, 30.0],
        },
        {
            "model": [RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)],
            "model__n_estimators": [160, 260],
            "model__max_depth": [8, 14, None],
            "model__min_samples_leaf": [1, 3],
        },
        {
            "model": [HistGradientBoostingRegressor(random_state=random_state)],
            "model__learning_rate": [0.04, 0.08],
            "model__max_iter": [180, 260],
            "model__max_leaf_nodes": [15, 31],
            "model__l2_regularization": [0.0, 0.1],
        },
    ]

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
    )
