"""
src/query_agent.py
------------------
Intent extraction via OpenAI structured output (Pydantic),
then live ECB fetch + load_and_predict + upsert into vector store.
"""

import json
import logging
import requests
from typing import List
from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from .BayesMolchanov import load_and_predict
from .vector_store import FinancialVectorStore

logger = logging.getLogger(__name__)

VECTOR_STORE_DIR  = "vector_store"
REGISTRY_PATH     = "models/registry.json"

# ---- series registry ----

SERIES_REGISTRY = {
    "ECB-FX-001": ("EXR", "D.USD.EUR.SP00.A", "Foreign Exchange", "Daily USD/EUR spot exchange rate"),
    "ECB-FX-002": ("EXR", "D.GBP.EUR.SP00.A", "Foreign Exchange", "Daily GBP/EUR spot exchange rate"),
    "ECB-FX-003": ("EXR", "D.JPY.EUR.SP00.A", "Foreign Exchange", "Daily JPY/EUR spot exchange rate"),
    "ECB-FX-004": ("EXR", "D.CHF.EUR.SP00.A", "Foreign Exchange", "Daily CHF/EUR spot exchange rate"),
}

SERIES_DESCRIPTIONS = "\n".join(
    f"  {rid}: {meta[3]}" for rid, meta in SERIES_REGISTRY.items()
)

# ---- Pydantic schema ----

class ECBQueryIntent(BaseModel):
    """Structured intent extracted from a user query about ECB data."""

    series_ids: List[str] = Field(
        description=(
            "List of ECB report IDs relevant to the query. "
            f"Choose from:\n{SERIES_DESCRIPTIONS}\n"
            "Return all IDs if the query is general or ambiguous."
        )
    )
    days: int = Field(
        ge=30,
        le=700,
        description=(
            "Number of days of history to fetch. "
            "Use 365 for a full year if unclear, 30 for last month."
        ),
    )


# ---- intent extractor ----

_llm            = ChatOpenAI(model="gpt-5.5", temperature=0)
_structured_llm = _llm.with_structured_output(ECBQueryIntent)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a financial data assistant. "
        "Given a user query about ECB exchange rates, "
        "identify which series are relevant and how many days of history are needed.",
    ),
    ("human", "{query}"),
])

_intent_chain = _prompt | _structured_llm


def extract_intent(query: str) -> ECBQueryIntent:
    try:
        intent = _intent_chain.invoke({"query": query})
        intent.series_ids = [s for s in intent.series_ids if s in SERIES_REGISTRY]
        if not intent.series_ids:
            intent.series_ids = list(SERIES_REGISTRY.keys())
        logger.info(f"Intent: series={intent.series_ids}, days={intent.days}")
        return intent
    except Exception as e:
        logger.warning(f"Intent extraction failed ({e}), using defaults.")
        return ECBQueryIntent(series_ids=list(SERIES_REGISTRY.keys()), days=365)


# ---- ECB fetch ----

def _fetch_ecb_series(series_key: str, dataset: str, last_n: int) -> dict | None:
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

        # last 5 only for LLM context
        key_statistics = [
            f"{all_periods[i]}: {round(values[i], 4)}"
            for i in range(max(0, len(values) - 5), len(values))
        ]

        latest_value  = values[-1]
        latest_period = all_periods[-1]

        # load pretrained model and run inference
        trend_stats = load_and_predict(
            series_id=report_id,
            values=values,
            n_future=forecast_days,
            registry_path=REGISTRY_PATH,
        )

        # build full date label list: observed + future
        last_date        = datetime.strptime(all_periods[-1], "%Y-%m-%d")
        future_dates     = [
            (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(forecast_days)
        ]
        full_date_labels = all_periods + future_dates

        # load model metadata from registry for RAG context
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
        entry = registry.get(report_id, {})

        model_info = {
            "hidden":        entry.get("hidden", 64),
            "activation":    entry.get("activation", "sin"),
            "warmup_epochs": entry.get("warmup_epochs", 150),
            "anneal_epochs": entry.get("anneal_epochs", 150),
            "lr_warmup":     entry.get("lr_warmup", 0.1),
            "lr_anneal":     entry.get("lr_anneal", 0.01),
            "final_loss":    entry.get("final_loss", "N/A"),
            "n_obs":         entry.get("n_obs", "N/A"),
        }

        return {
            "reportId":         report_id,
            "title":            f"{title_full} — {latest_period}",
            "author":           "European Central Bank",
            "publication_date": datetime.now().strftime("%Y-%m-%d"),
            "category":         category,
            "currency_pair":    f"{currency}/{denom}",
            "latest_value":     round(latest_value, 4),
            "latest_period":    latest_period,
            "summary": (
                f"{description}. Latest: {round(latest_value, 4)} "
                f"({currency}/{denom}) for {latest_period}. "
                f"Bayesian VDM MLP [1→{model_info['hidden']}→1] trained on "
                f"{entry.get('trained_on', 'N/A')} ({model_info['n_obs']} obs). "
                f"Forecast horizon: {forecast_days} days. "
                f"Posterior predictive: {trend_stats.get('n_samples', 200)} samples."
            ),
            "key_statistics":  key_statistics,
            "date_labels":     full_date_labels,
            "model_info":      model_info,
            "trend":           trend_stats,
        }
    except Exception as e:
        logger.error(f"Failed to build report for {report_id}: {e}")
        return None


# ---- document builder ----

def _report_to_document(report: dict) -> Document:
    trend      = report.get("trend", {})
    model_info = report.get("model_info", {})
    content = (
        f"Title: {report['title']}\n"
        f"Category: {report['category']}\n"
        f"Currency pair: {report['currency_pair']}\n"
        f"Latest value: {report['latest_value']} for {report['latest_period']}\n"
        f"Summary: {report['summary']}\n"
        f"Model architecture: [1 → {model_info.get('hidden', 64)} → 1] "
        f"activation={model_info.get('activation', 'sin')}\n"
        f"Training: {model_info.get('warmup_epochs', 150)} warmup + "
        f"{model_info.get('anneal_epochs', 150)} anneal epochs\n"
        f"Learning rates: lr_warmup={model_info.get('lr_warmup', 0.1)} "
        f"lr_anneal={model_info.get('lr_anneal', 0.01)}\n"
        f"Final ELBO loss: {model_info.get('final_loss', 'N/A')}\n"
        f"Trained on N obs: {model_info.get('n_obs', 'N/A')}\n"
        f"Model trained on date: {trend.get('trained_on', 'N/A')}\n"
        f"Forecast horizon: {trend.get('n_future', 30)} days\n"
        f"Posterior samples: {trend.get('n_samples', 200)}\n"
        f"Key statistics (last 5 days):\n"
        + "\n".join(report.get("key_statistics", []))
    )
    return Document(
        page_content=content,
        metadata={
            "report_id":        report["reportId"],
            "title":            report["title"],
            "category":         report["category"],
            "publication_date": report["publication_date"],
            "author":           report["author"],
            "source":           "ECB Data API",
        },
    )


# ---- public entry point ----

def fetch_and_upsert(query: str,
                     vector_store: FinancialVectorStore,
                     save_path: str = VECTOR_STORE_DIR,
                     forecast_days: int = 30) -> dict:
    """
    Extract intent → fetch ECB data → load pretrained Bayesian MLP
    → upsert vector store. Reports always returned for chart rendering.
    """
    intent  = extract_intent(query)
    reports = []

    for report_id in intent.series_ids:
        dataset, series_key, category, description = SERIES_REGISTRY[report_id]
        raw = _fetch_ecb_series(series_key, dataset, 365)
        if raw is None:
            continue
        report = _build_report(raw, report_id, category, description,
                               forecast_days=forecast_days)
        if report is None:
            continue
        reports.append(report)

    documents    = [_report_to_document(r) for r in reports]
    upsert_stats = vector_store.upsert_documents(documents, save_path=save_path)

    return {
        "intent":  intent,
        "added":   upsert_stats["added"],
        "skipped": upsert_stats["skipped"],
        "reports": reports,
    }