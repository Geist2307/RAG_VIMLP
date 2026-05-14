import pytest
from langchain_core.documents import Document
from src.rag.document_loader import ECBSpeechLoader


class TestECBSpeechLoader:

    def test_load_reports(self, reports_json_file, sample_ecb_speeches):
        """Test loading speeches from temporary JSON file."""
        loader = ECBSpeechLoader(reports_json_file)
        loaded = loader.load_reports()

        assert len(loaded) == 4
        assert loaded[0]["metadata"]["speaker"] == "Philip R. Lane"
        assert loaded[1]["metadata"]["speaker"] == "Piero Cipollone"
        assert loaded[2]["metadata"]["speaker"] == "Piero Cipollone"
        assert loaded[3]["metadata"]["speaker"] == "Frank Elderson"

    def test_create_documents(self, reports_json_file):
        """Test LangChain Document creation from speeches."""
        loader = ECBSpeechLoader(reports_json_file)
        documents = loader.create_documents()

        assert len(documents) == 4
        assert all(isinstance(doc, Document) for doc in documents)

        doc1 = documents[0]

        # check structure matches ECBSpeechLoader format
        assert "Philip R. Lane" in doc1.page_content
        assert "Title:" in doc1.page_content
        assert "Speaker:" in doc1.page_content
        assert "Publication Date:" in doc1.page_content
        assert doc1.metadata["source"] == "ECB Speeches Dataset"
        assert doc1.metadata["speaker"] == "Philip R. Lane"

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        loader = ECBSpeechLoader("nonexistent.json")
        with pytest.raises(FileNotFoundError):
            loader.load_reports()

    def test_invalid_json(self, tmp_path):
        """Test handling of invalid JSON file."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("invalid json content")

        loader = ECBSpeechLoader(str(invalid_file))
        with pytest.raises(ValueError):
            loader.load_reports()