"""
get_reports.py
--------------
Downloads ECB speeches dataset directly from the ECB website (no scraping),
filters to the last two years, and saves as structured JSON for the RAG pipeline.

Each speech becomes one LangChain Document in the vector store.

Run:
    python get_reports.py
Output:
    data/ecb_speeches.json
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO

SPEECHES_URL = "https://www.ecb.europa.eu/press/key/shared/data/all_ECB_speeches.csv"
OUTPUT_PATH  = "data/ecb_speeches.json"
CUTOFF_YEARS = 2


def fetch_speeches() -> pd.DataFrame:
    """Download speeches CSV directly from ECB website."""
    print("Fetching ECB speeches dataset...")
    response = requests.get(SPEECHES_URL, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text), sep="|", encoding="utf-8")
    print(f"✓ Downloaded {len(df)} speeches total")
    return df


def filter_recent(df: pd.DataFrame, years: int = CUTOFF_YEARS) -> pd.DataFrame:
    """Keep only speeches from the last N years."""
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    cutoff = datetime.now() - timedelta(days=years * 365)
    recent = df[df["date"] >= cutoff].copy()
    print(f"✓ Filtered to {len(recent)} speeches from last {years} years")
    return recent


def to_documents(df: pd.DataFrame) -> list[dict]:
    """
    Convert each row to a document dict matching LangChain Document structure.
    page_content = full speech text
    metadata     = date, speaker, title, subtitle, source
    """
    documents = []
    skipped   = 0

    for _, row in df.iterrows():
        content = str(row.get("contents", "")).strip()
        if not content or content == "nan":
            skipped += 1
            continue

        documents.append({
            "page_content": content,
            "metadata": {
                "date":     str(row.get("date", ""))[:10],   # YYYY-MM-DD
                "speaker":  str(row.get("speakers", "N/A")),
                "title":    str(row.get("title", "N/A")),
                "subtitle": str(row.get("subtitle", "")),
                "source":   "ECB Speeches Dataset",
            }
        })

    print(f"✓ Built {len(documents)} documents ({skipped} skipped — no content)")
    return documents


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df        = fetch_speeches()
    df        = filter_recent(df, years=CUTOFF_YEARS)
    documents = to_documents(df)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(documents)} speech documents → {OUTPUT_PATH}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")