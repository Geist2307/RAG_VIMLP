# macOS: PyTorch and faiss-cpu both link OpenMP; without this, FAISS search can abort the process.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import tempfile
from pathlib import Path
import pytest

# we start with a fixture 
@pytest.fixture
def sample_financial_reports():
    """Sample financial report data for testing"""
    return [
  {
    "reportId": "ECB-FX-001",
    "title": "US dollar/Euro ECB reference exchange rate \u2014 2026-04",
    "author": "European Central Bank",
    "publication_date": "2026-05-06",
    "category": "Foreign Exchange",
    "summary": "Monthly average USD/EUR spot exchange rate. Latest observation: 1.1706 (US dollar/Euro) for 2026-04. Trend over last 24 months: upward.",
    "key_statistics": [
      "2024-05: 1.0812",
      "2024-06: 1.0759",
      "2024-07: 1.0844",
      "2024-08: 1.1012",
      "2024-09: 1.1106",
      "2024-10: 1.0904",
      "2024-11: 1.063",
      "2024-12: 1.0479",
      "2025-01: 1.0354",
      "2025-02: 1.0413",
      "2025-03: 1.0807",
      "2025-04: 1.1214",
      "2025-05: 1.1278",
      "2025-06: 1.1516",
      "2025-07: 1.1677",
      "2025-08: 1.1631",
      "2025-09: 1.1732",
      "2025-10: 1.163",
      "2025-11: 1.156",
      "2025-12: 1.1709",
      "2026-01: 1.1738",
      "2026-02: 1.1824",
      "2026-03: 1.1558",
      "2026-04: 1.1706"
    ],
    "sentiment_score": 0.5128
  },
  {
    "reportId": "ECB-FX-002",
    "title": "Pound sterling/Euro ECB reference exchange rate \u2014 2026-04",
    "author": "European Central Bank",
    "publication_date": "2026-05-06",
    "category": "Foreign Exchange",
    "summary": "Monthly average GBP/EUR spot exchange rate. Latest observation: 0.8693 (UK pound sterling/Euro) for 2026-04. Trend over last 24 months: upward.",
    "key_statistics": [
      "2024-05: 0.8556",
      "2024-06: 0.8464",
      "2024-07: 0.8433",
      "2024-08: 0.8515",
      "2024-09: 0.8402",
      "2024-10: 0.835",
      "2024-11: 0.8338",
      "2024-12: 0.828",
      "2025-01: 0.8391",
      "2025-02: 0.8307",
      "2025-03: 0.837",
      "2025-04: 0.8538",
      "2025-05: 0.8435",
      "2025-06: 0.8498",
      "2025-07: 0.8647",
      "2025-08: 0.8653",
      "2025-09: 0.8689",
      "2025-10: 0.8716",
      "2025-11: 0.88",
      "2025-12: 0.875",
      "2026-01: 0.8683",
      "2026-02: 0.8703",
      "2026-03: 0.8663",
      "2026-04: 0.8693"
    ],
    "sentiment_score": 0.5035
  }

    ]

@pytest.fixture
def reports_json_file(sample_financial_reports):
    """Create a temporary financial reports JSON file"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(sample_financial_reports, f, indent=2)
        tmp_path = f.name
    yield tmp_path
    os.unlink(tmp_path) # we cleanup after the test


@pytest.fixture
def test_environment(sample_financial_reports):
    """Create test environment with financial reports data"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        reports_file = tmp_path / "ecb_reports.json"

        with open(reports_file, "w") as f:
            json.dump(sample_financial_reports, f, indent=2)

        yield {"tmp_dir": tmp_dir, "reports_file": str(reports_file)} 