import pytest
import re
from langchain_core.documents import Document

from src.document_loader import FinancialDocumentLoader

class TestFinancialDocumentLoader:
    def test_load_reports(self, reports_json_file, sample_financial_reports):
        """Test loading financial reports from JSON file"""
        loader = FinancialDocumentLoader(reports_json_file)
        loaded_reports = loader.load_reports()

        assert loaded_reports == sample_financial_reports
        assert len(loaded_reports) == 2 
        assert loaded_reports[0]["reportId"] == "ECB-FX-001"
        assert loaded_reports[1]["reportId"] == "ECB-FX-002"

    def test_create_documents(self, reports_json_file):
        """Test create langchain documnets from report files"""
        loader = FinancialDocumentLoader(reports_json_file)
        documents = loader.create_documents()

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)

        # Test US dollar/ EUR document structure
        us_eur_doc = documents[0]

        # let's test key words in content
        assert "Monthly operations for" in us_eur_doc.page_content
        assert "Author: European Central Bank" in us_eur_doc.page_content
        assert "Sentiment Score: 0.5128" in us_eur_doc.page_content
        assert us_eur_doc.metadata["author"] == "European Central Bank"
        assert us_eur_doc.metadata["report_id"] == "ECB-FX-001"

    

        # Test GBP/EUR report
        gbp_eur_report = documents[1]

        # let's test key workds in content
        assert "Monthly operations for" in gbp_eur_report.page_content
        assert "Author: European Central Bank" in gbp_eur_report.page_content
        assert "Sentiment Score: 0.5035" in gbp_eur_report.page_content
        assert gbp_eur_report.metadata["author"] == "European Central Bank"
        assert gbp_eur_report.metadata["report_id"] == "ECB-FX-002"

    def test_file_not_found(self):
        """Test handling of non-existent file"""

        loader = FinancialDocumentLoader("nonexistent.json")
        with pytest.raises(FileNotFoundError):
            loader.load_reports()

    def test_invalid_json(self, tmp_path):
        """Test handling of invalid JSON file"""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("invalid json content")

        loader = FinancialDocumentLoader(str(invalid_file))
        with pytest.raises(ValueError):
            loader.load_reports() 