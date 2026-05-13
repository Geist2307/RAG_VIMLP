"""
src/trend_enricher.py
---------------------
Injects factual ECB context into the RAG prompt.
Reads model metadata directly from the registry by series ID.
Extracts forecast predictions and uncertainty from live trend stats.
"""

import json
import numpy as np
from pathlib import Path

REGISTRY_PATH = Path("models/registry.json")


def _load_registry(registry_path: Path = REGISTRY_PATH) -> dict:
    with open(registry_path) as f:
        return json.load(f)


def _extract_forecast_summary(trend: dict, entry: dict) -> list[str]:
    """
    Extract representative forecast checkpoints from the posterior predictive.
    Returns a list of strings like:
      "2026-05-13: mean=1.0832, σ=0.0023"
    """
    y_mean      = trend.get("y_mean", [])
    y_std       = trend.get("y_std", [])
    date_labels = trend.get("date_labels", [])
    n_future    = trend.get("n_future")
    n_obs       = entry.get("n_obs")

    if not y_mean or not n_future or not n_obs:
        return ["Forecast data unavailable."]

    total_grid = len(y_mean)
    split      = int(total_grid * n_obs / (n_obs + n_future))

    # forecast portion only
    forecast_mean   = y_mean[split:]
    forecast_std    = y_std[split:]
    forecast_dates  = date_labels[n_obs:] if len(date_labels) > n_obs else []
    forecast_len    = len(forecast_mean)

    if forecast_len == 0:
        return ["Forecast data unavailable."]

    # representative checkpoints: day 1, day 7, midpoint, last day
    checkpoints = sorted(set([
        0,
        min(6, forecast_len - 1),
        forecast_len // 2,
        forecast_len - 1,
    ]))

    lines = []
    for i in checkpoints:
        date  = forecast_dates[i] if i < len(forecast_dates) else f"+{i+1}d"
        mean  = round(float(forecast_mean[i]), 4)
        sigma = round(float(forecast_std[i]),  4)
        lines.append(f"    {date}: mean={mean}, σ={sigma}")

    return lines


def format_fact_context(reports: list[dict],
                        registry_path: Path = REGISTRY_PATH) -> str:
    """
    Render factual ECB data as a labelled string for the LLM.

    For each report:
      - Reads model metadata from registry by series ID
      - Includes last 5 market observations
      - Extracts forecast checkpoints + uncertainty from live trend stats
    """
    if not reports:
        return ""

    registry = _load_registry(registry_path)
    lines    = ["[ECB EXCHANGE RATE DATA — Source: ECB Data API]"]

    for r in reports:
        report_id  = r.get("reportId")
        trend      = r.get("trend", {})
        key_stats  = r.get("key_statistics", [])

        # load model metadata from registry — single source of truth
        entry = registry.get(report_id, {})
        if not entry:
            lines.append(f"\n• {report_id} — no registry entry found.\n")
            continue

        n_obs = entry.get("n_obs")
        if n_obs is None:
            raise ValueError(f"n_obs missing from registry for {report_id}")

        forecast_lines = _extract_forecast_summary(trend, entry)

        lines.append(
            f"\n• {r.get('title', 'N/A')} ({report_id})\n"

            f"\n  — MARKET DATA —\n"
            f"  Currency pair    : {r.get('currency_pair', 'N/A')}\n"
            f"  Latest available : {r.get('latest_value', 'N/A')} "
            f"as of {r.get('latest_period', 'N/A')}\n"
            f"  Last 5 observations:\n"
            + "\n".join(f"    {s}" for s in key_stats) +

            f"\n\n  — BAYESIAN MODEL —\n"
            f"  Architecture     : [1 → {entry['hidden']} → 1] "
            f"activation={entry['activation']}\n"
            f"  Trained on       : {entry['trained_on']} "
            f"({n_obs} observations)\n"
            f"  Training schedule: {entry['warmup_epochs']} warmup + "
            f"{entry['anneal_epochs']} annealing epochs\n"
            f"  Learning rates   : warmup={entry['lr_warmup']} "
            f"anneal={entry['lr_anneal']}\n"
            f"  Final ELBO loss  : {entry['final_loss']}\n"
            f"  Method           : Variational Dropout (Molchanov et al. 2017)\n"

            f"\n  — FORECAST (next {trend.get('n_future', '?')} days) —\n"
            f"  Posterior samples: {trend.get('n_samples', 200)}\n"
            f"  Note: uncertainty (σ) widens with horizon as expected.\n"
            f"  Checkpoints (mean ± σ in original units):\n"
            + "\n".join(forecast_lines)
        )

    return "\n".join(lines)


class TrendEnricher:
    """
    Callable — accepts live reports and returns formatted context
    for the RAG prompt. Reads model metadata from registry.
    """

    def __init__(self, registry_path: Path = REGISTRY_PATH):
        self.registry_path = registry_path

    def __call__(self, reports: list[dict]) -> str:
        return format_fact_context(reports, registry_path=self.registry_path)