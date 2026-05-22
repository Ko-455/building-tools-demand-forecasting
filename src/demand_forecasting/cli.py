from __future__ import annotations

import argparse
import json

from demand_forecasting.config import load_config
from demand_forecasting.monitor import build_monitoring_report
from demand_forecasting.predict import predict_last_period
from demand_forecasting.train import train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Demand forecasting ML pipeline")
    parser.add_argument("command", choices=["train", "monitor", "predict"])
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--model-path", default="models/demand_forecast_model.joblib")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.command == "train":
        print(json.dumps(train_pipeline(config), indent=2, ensure_ascii=False))
    elif args.command == "monitor":
        print(json.dumps(build_monitoring_report(config), indent=2, ensure_ascii=False))
    elif args.command == "predict":
        print(predict_last_period(config, model_path=args.model_path).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
