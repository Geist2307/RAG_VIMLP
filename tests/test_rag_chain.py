import pytest
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv
from langchain_core.documents import Document
from src.rag_chain import FinancialRAGChain
from src.vector_store import FinancialVectorStore
from src.document_loader import FinancialDocumentLoader

load_dotenv()

class TestFinancialRAGChain:

    @pytest.fixture
    def vector_store(self, reports_json_file):
        """Create a real vector store from ECB documents"""
        loader = FinancialDocumentLoader(reports_json_file)
        documents = loader.create_documents()
        store = FinancialVectorStore()
        store.create_vector_store(documents)
        return store

    @pytest.fixture
    def rag_chain(self, vector_store):
        """Create RAG chain with real vector store and real LLM"""
        return FinancialRAGChain(vector_store)

    # --- validation tests, no mocking needed ---

    def test_empty_query_raises(self, rag_chain):
        """Empty query should raise ValueError"""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            rag_chain.query("")

    def test_whitespace_query_raises(self, rag_chain):
        """Whitespace only query should raise ValueError"""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            rag_chain.query("   ")

    def test_short_query_raises(self, rag_chain):
        """Query shorter than 10 characters should raise ValueError"""
        with pytest.raises(ValueError, match="Query is to short"):
            rag_chain.query("USD")

    # --- get_relevant_documents tests ---

    def test_get_relevant_documents(self, rag_chain):
        """Test retrieval returns correct structure"""
        results = rag_chain.get_relevant_documents("US dollar Euro exchange rate", k=1)

        assert isinstance(results, list)
        assert len(results) == 1

        # check structure of each result
        assert "content" in results[0]
        assert "metadata" in results[0]

        # check content is relevant
        assert "USD" in results[0]["content"] or "US dollar" in results[0]["content"]

    def test_get_relevant_documents_empty_query(self, rag_chain):
        """Empty query should raise ValueError in get_relevant_documents"""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            rag_chain.get_relevant_documents("")

    def test_get_relevant_documents_short_query(self, rag_chain):
        """Short query should raise ValueError in get_relevant_documents"""
        with pytest.raises(ValueError, match="Query is to short"):
            rag_chain.get_relevant_documents("GBP")

    # --- chain tests ---

    def test_chain_lazy_initialization(self, rag_chain):
        """Chain should be None until first query"""
        assert rag_chain.chain is None
        rag_chain._get_chain()
        assert rag_chain.chain is not None

    def test_query_returns_string(self, rag_chain):
        """Query should return a non-empty string response"""
        result = rag_chain.query("What is the USD EUR exchange rate trend?")
        assert isinstance(result, str)
        assert len(result) > 0