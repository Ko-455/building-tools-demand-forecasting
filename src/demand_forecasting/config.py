from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    sales_path: Path
    price_list_path: Path
    product_type: str
    date_column: str
    target_column: str
    forecast_horizon_weeks: int


@dataclass(frozen=True)
class TrainingConfig:
    random_state: int
    cv_splits: int
    scoring: str
    n_jobs: int


@dataclass(frozen=True)
class MonitoringConfig:
    psi_warning_threshold: float
    psi_critical_threshold: float
    max_missing_share: float


@dataclass(frozen=True)
class ProjectConfig:
    title: str
    github_url: str


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    data: DataConfig
    training: TrainingConfig
    monitoring: MonitoringConfig


def load_config(path: str | Path = "config.yml") -> Config:
    config_path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent

    data = raw["data"]
    return Config(
        project=ProjectConfig(**raw["project"]),
        data=DataConfig(
            sales_path=(base / data["sales_path"]).resolve(),
            price_list_path=(base / data["price_list_path"]).resolve(),
            product_type=data["product_type"],
            date_column=data["date_column"],
            target_column=data["target_column"],
            forecast_horizon_weeks=int(data["forecast_horizon_weeks"]),
        ),
        training=TrainingConfig(**raw["training"]),
        monitoring=MonitoringConfig(**raw["monitoring"]),
    )
