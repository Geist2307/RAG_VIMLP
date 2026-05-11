"""
src/trend_enricher.py
---------------------
Injects factual ECB context into the RAG prompt.
No trend statistics — the chart handles that visually.
The agent cites facts only: latest value, date, currency pair, model info.
"""

import json
from pathlib import Path

DATA_PATH = Path("data/ecb_reports.json")

KEYWORD_MAP = {
    "usd":      ["ECB-FX-001"],
    "dollar":   ["ECB-FX-001"],
    "gbp":      ["ECB-FX-002"],
    "pound":    ["ECB-FX-002"],
    "jpy":      ["ECB-FX-003"],
    "yen":      ["ECB-FX-003"],
    "chf":      ["ECB-FX-004"],
    "franc":    ["ECB-FX-004"],
    "fx":       ["ECB-FX-001", "ECB-FX-002", "ECB-FX-003", "ECB-FX-004"],
    "exchange": ["ECB-FX-001", "ECB-FX-002", "ECB-FX-003", "ECB-FX-004"],
}


def _resolve_report_ids(query: str) -> list[str] | None:
    q = query.lower()
    matched = set()
    for keyword, ids in KEYWORD_MAP.items():
        if keyword in q:
            matched.update(ids)
    return list(matched) if matched else None


def format_fact_context(reports: list[dict]) -> str:
    """
    Render factual ECB data as a labelled string for the LLM.
    Only cites what the data actually contains — no trend inference.
    """
    if not reports:
        return ""

    lines = ["[ECB EXCHANGE RATE DATA — Source: ECB Data API]"]
    for r in reports:
        trend = r.get("trend", {})
        key_stats = r.get("key_statistics", [])
        lines.append(
            f"\n• {r.get('title', 'N/A')} ({r.get('reportId', 'N/A')})\n"
            f"  Currency pair  : {r.get('currency_pair', 'N/A')}\n"
            f"  Latest value   : {r.get('latest_value', 'N/A')} "
            f"for {r.get('latest_period', 'N/A')}\n"
            f"  Model trained  : {trend.get('trained_on', 'N/A')}\n"
            f"  Forecast horizon: {trend.get('n_future', 30)} days\n"
            f"  Posterior samples: {trend.get('n_samples', 200)}\n"
            f"  Last 5 observations:\n"
            + "\n".join(f"    {s}" for s in key_stats)
        )
    return "\n".join(lines)


class TrendEnricher:
    """
    Callable — accepts a query string, returns formatted factual context
    ready to be merged with the retriever output in rag_chain.py.
    """

    def __init__(self, data_path: Path = DATA_PATH):
        self.data_path = data_path
        self._cache: list[dict] | None = None

    def _load(self) -> list[dict]:
        if self._cache is None:
            with open(self.data_path) as f:
                self._cache = json.load(f)
        return self._cache

    def __call__(self, query: str) -> str:
        reports = self._load()
        ids = _resolve_report_ids(query)
        if ids is not None:
            reports = [r for r in reports if r.get("reportId") in ids]
        return format_fact_context(reports)