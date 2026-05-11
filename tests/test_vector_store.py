import pytest
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from src.vector_store import FinancialVectorStore
from src.document_loader import FinancialDocumentLoader

load_dotenv()

class TestFinancialVectorStore:

    @pytest.fixture
    def sample_documents(self, reports_json_file):
        """Create ECB documents using the actual loader"""
        loader = FinancialDocumentLoader(reports_json_file)
        return loader.create_documents()

    def test_create_vector_store(self, sample_documents):
        """Test creating vector store from ECB documents"""
        store = FinancialVectorStore()
        store.create_vector_store(sample_documents)
        assert store.vector_store is not None

    def test_similarity_search(self, sample_documents):
        """Test similarity search with ECB relevant queries"""
        store = FinancialVectorStore()
        store.create_vector_store(sample_documents)

        # query relevant to ECB data
        results = store.similarity_search("US dollar Euro exchange rate", k=1)
        assert len(results) == 1
        assert "USD" in results[0].page_content or "US dollar" in results[0].page_content

    def test_similarity_search_without_initialization(self):
        """Test error handling when searching without initialization"""
        store = FinancialVectorStore()
        with pytest.raises(ValueError):
            store.similarity_search("euro exchange rate")

    def test_empty_query(self, sample_documents):
        """Test handling of empty query"""
        store = FinancialVectorStore()
        store.create_vector_store(sample_documents)
        results = store.similarity_search("")
        assert isinstance(results, list)
        assert len(results) == 0

    def test_short_query(self, sample_documents):
        """Test handling of query that is too short"""
        store = FinancialVectorStore()
        store.create_vector_store(sample_documents)
        results = store.similarity_search("USD")
        assert isinstance(results, list)
        assert len(results) == 0

    def test_save_and_load_local(self, sample_documents, tmp_path):
        """Test saving and loading vector store locally"""
        store = FinancialVectorStore()
        store.create_vector_store(sample_documents)
        save_path = str(tmp_path / "vector_store")
        store.save_local(save_path)

        loaded_store = FinancialVectorStore.load_local(save_path)
        assert loaded_store.vector_store is not None

        results = loaded_store.similarity_search("pound sterling euro exchange rate", k=1)
        assert len(results) == 1
        assert "Pound sterling" in results[0].page_content or "GBP" in results[0].page_content