"""
src/data/orchestrator.py
------------------------
Orchestrates the data pipeline for the ECB FX analysis app.

Responsibilities:
    - Read series configuration from models/registry.json
    - Fetch live ECB FX data from SDMX API
    - Parse ECB API response into structured report
    - Run pretrained Bayesian MLP inference

No LLM calls — series selection handled by UI in app.py.
No vector store updates — speeches managed separately by get_reports.py.
registry.json is the single source of truth for all series configuration.
"""

import json
import logging
import requests
from datetime import datetime, timedelta

from src.bayesianMLP.inference import load_and_predict

logger = logging.getLogger(__name__)

REGISTRY_PATH = "models/registry.json"


# ---- ECB API fetch ----

def _fetch_ecb_series(series_key: str, dataset: str, last_n: int = 365) -> dict | None:
    """Fetch raw ECB SDMX data for one series."""
    url = f"https://data-api.ecb.europa.eu/service/data/{dataset}/{series_key}"
    try:
        r = requests.get(
            url,
            params={"lastNObservations": last_n, "format": "jsondata"},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"ECB API error for {dataset}/{series_key}: {e}")
        return None


# ---- report builder ----

def _build_report(data: dict, report_id: str, category: str,
                  description: str, forecast_days: int = 30) -> dict | None:
    """
    Parse ECB API response and run inference.
    No registry access here — Enricher owns all registry metadata.

    Returns a live report dict with:
        reportId, title, currency_pair, latest_value, latest_period,
        key_statistics, date_labels, trend (from load_and_predict)
    """
    try:
        structure         = data["structure"]
        series_key_actual = list(data["dataSets"][0]["series"].keys())[0]
        series            = data["dataSets"][0]["series"][series_key_actual]
        time_periods      = structure["dimensions"]["observation"][0]["values"]
        observations      = series["observations"]

        dims     = structure["dimensions"]["series"]
        currency = dims[1]["values"][0]["name"] if len(dims) > 1 and dims[1]["values"] else "N/A"
        denom    = dims[2]["values"][0]["name"] if len(dims) > 2 and dims[2]["values"] else "N/A"

        attrs      = structure["attributes"]["series"]
        title_attr = next((a for a in attrs if a.get("name") == "Title"), None)
        title_full = (title_attr["values"][0]["name"]
                      if title_attr and title_attr["values"] else "ECB Indicator")

        # collect all values and dates
        values, all_periods = [], []
        for idx, period in enumerate(time_periods):
            value = observations.get(str(idx), [None])[0]
            if value is not None:
                values.append(value)
                all_periods.append(period["name"])

        if not values:
            return None

        latest_value  = values[-1]
        latest_period = all_periods[-1]

        # last 5 only for LLM context
        key_statistics = [
            f"{all_periods[i]}: {round(values[i], 4)}"
            for i in range(max(0, len(values) - 5), len(values))
        ]

        # full date labels: observed + forecast
        last_date        = datetime.strptime(all_periods[-1], "%Y-%m-%d")
        future_dates     = [
            (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(forecast_days)
        ]
        full_date_labels = all_periods + future_dates

        # inference — no registry access here
        trend_stats = load_and_predict(
            series_id=report_id,
            values=values,
            n_future=forecast_days,
            registry_path=REGISTRY_PATH,
        )
        trend_stats["date_labels"] = full_date_labels

        return {
            "reportId":         report_id,
            "title":            f"{title_full} — {latest_period}",
            "author":           "European Central Bank",
            "publication_date": datetime.now().strftime("%Y-%m-%d"),
            "category":         category,
            "currency_pair":    f"{currency}/{denom}",
            "latest_value":     round(latest_value, 4),
            "latest_period":    latest_period,
            "key_statistics":   key_statistics,
            "date_labels":      full_date_labels,
            "trend":            trend_stats,
        }

    except Exception as e:
        logger.error(f"Failed to build report for {report_id}: {e}")
        return None


# ---- public entry point ----

def fetch_reports(series_ids: list[str],
                  forecast_days: int = 30) -> list[dict]:
    """
    Fetch ECB data and run inference for selected series.
    Series configuration is read from registry.json — no hardcoded series list.

    Args:
        series_ids:    list of series IDs selected from UI
        forecast_days: forecast horizon from UI slider

    Returns:
        list of live report dicts for chart rendering and RAG context
    """
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    reports = []

    for series_id in series_ids:
        entry = registry.get(series_id)
        if not entry:
            logger.warning(f"Unknown series_id: {series_id} — not in registry")
            continue

        raw = _fetch_ecb_series(
            entry["series_key"],
            entry["dataset"],
            last_n=entry.get("n_obs", 365),
        )
        if raw is None:
            continue

        report = _build_report(
            raw, series_id,
            entry["category"],
            entry["description"],
            forecast_days=forecast_days,
        )
        if report:
            reports.append(report)
            logger.info(f"✓ Built report for {series_id}")

    return reports