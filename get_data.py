import requests
import json
from datetime import datetime
import os

# automatically extract for last year
def fetch_ecb_series(series_key, dataset="EXR", last_n=365):
    url = f"https://data-api.ecb.europa.eu/service/data/{dataset}/{series_key}"
    params = {"lastNObservations": last_n, "format": "jsondata"}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def transform_to_report_schema(data, report_id, category, description, last_n=365):
    structure = data["structure"]

    series_key_actual = list(data["dataSets"][0]["series"].keys())[0]
    series = data["dataSets"][0]["series"][series_key_actual]

    time_periods = structure["dimensions"]["observation"][0]["values"]
    observations = series["observations"]

    dims = structure["dimensions"]["series"]
    currency = dims[1]["values"][0]["name"] if len(dims) > 1 and dims[1]["values"] else "N/A"
    denom    = dims[2]["values"][0]["name"] if len(dims) > 2 and dims[2]["values"] else "N/A"

    attrs = structure["attributes"]["series"]
    title_attr = next((a for a in attrs if a.get("name") == "Title"), None)
    title_full = (title_attr["values"][0]["name"]
                  if title_attr and title_attr["values"]
                  else "ECB Economic Indicator")

    # --- collect ordered values ---
# collect values
    values = []
    all_periods = []
    for idx, period in enumerate(time_periods):
        value = observations.get(str(idx), [None])[0]
        if value is not None:
            values.append(value)
            all_periods.append(period["name"])

    latest_value  = values[-1] if values else None
    latest_period = all_periods[-1] if all_periods else "N/A"

    # only last 5 observations for LLM context
    key_statistics = [
        f"{all_periods[i]}: {round(values[i], 4)}"
        for i in range(max(0, len(values) - 5), len(values))
    ]

    return {
        "reportId":          report_id,
        "title":             f"{title_full} — {latest_period}",
        "author":            "European Central Bank",
        "publication_date":  datetime.now().strftime("%Y-%m-%d"),
        "category":          category,
        "currency_pair":     f"{currency}/{denom}",
        "latest_value":      round(latest_value, 4) if latest_value is not None else None,
        "latest_period":     latest_period,
        "summary": (
            f"{description}. Latest observation: {round(latest_value, 4)} "
            f"({currency}/{denom}) for {latest_period}. "
            
        ),
        "key_statistics":    key_statistics,
        "values" : values ## all 365
        }


# ---- series registry ----
series_configs = [
    ("EXR/D.USD.EUR.SP00.A", "ECB-FX-001", "Foreign Exchange",
     "Daily  USD/EUR spot exchange rate"),
    ("EXR/D.GBP.EUR.SP00.A", "ECB-FX-002", "Foreign Exchange",
     "Daily  GBP/EUR spot exchange rate"),
    ("EXR/D.JPY.EUR.SP00.A", "ECB-FX-003", "Foreign Exchange",
     "Daily  JPY/EUR spot exchange rate"),
    ("EXR/D.CHF.EUR.SP00.A", "ECB-FX-004", "Foreign Exchange",
     "Daily  CHF/EUR spot exchange rate"),
]

if __name__ == "__main__":
    reports = []
    for series_key, report_id, category, description in series_configs:
        try:
            dataset, series = series_key.split("/", 1)
            raw    = fetch_ecb_series(series, dataset=dataset, last_n=365)
            report = transform_to_report_schema(raw, report_id, category, description)
            reports.append(report)
           
            print(f"✓ {report_id}  latest={report['latest_value']}  period={report['latest_period']}")
        except Exception as e:
            print(f"✗ Failed {report_id}: {e}")

    os.makedirs("data", exist_ok=True)
    with open("data/ecb_reports.json", "w") as f:
        json.dump(reports, f, indent=2)
    print("\nSaved to data/ecb_reports.json")