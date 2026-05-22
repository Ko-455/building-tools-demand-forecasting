FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY tests ./tests
COPY config.yml ./
COPY building_supply_store_sales_2020-2024.csv ./
COPY data ./data

RUN mkdir -p models reports/figures outputs && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "demand_forecasting.cli", "train", "--config", "config.yml"]
