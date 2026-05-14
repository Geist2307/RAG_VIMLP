import pytest

from dotenv import load_dotenv
from langchain_core.documents import Document

from src.rag.chain import FinancialRAGChain
from src.rag.vector_store import FinancialVectorStore
from src.rag.document_loader import ECBSpeechLoader

load_dotenv()

@pytest.mark.integration
class TestFinancialRAGChain:

    @pytest.fixture
    def vector_store(self, reports_json_file):
        """Create a real vector store from ECB speech documents."""
        loader    = ECBSpeechLoader(reports_json_file)
        documents = loader.create_documents()
        store     = FinancialVectorStore()
        store.create_vector_store(documents)
        return store

    @pytest.fixture
    def rag_chain(self, vector_store):
        """Create RAG chain with real vector store."""
        return FinancialRAGChain(vector_store)

    # ── validation tests — no LLM calls needed ────────────────────

    def test_empty_query_raises(self, rag_chain):
        """Empty query should raise ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            rag_chain.query("", reports=[], style="balanced")

    def test_whitespace_query_raises(self, rag_chain):
        """Whitespace-only query should raise ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            rag_chain.query("   ", reports=[], style="balanced")

    def test_short_query_raises(self, rag_chain):
        """Query shorter than 10 characters should raise ValueError."""
        with pytest.raises(ValueError, match="Query is too short"):
            rag_chain.query("USD", reports=[], style="balanced")

    # ── get_relevant_documents tests ──────────────────────────────

    def test_get_relevant_documents(self, rag_chain):
        """Retrieval should return correct structure."""
        results = rag_chain.get_relevant_documents(
            "euro safe assets financial markets", k=1
        )
        assert isinstance(results, list)
        assert len(results) == 1
        assert "content"  in results[0]
        assert "metadata" in results[0]

    def test_get_relevant_documents_empty_query(self, rag_chain):
        """Empty query should raise ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            rag_chain.get_relevant_documents("")

    def test_get_relevant_documents_short_query(self, rag_chain):
        """Short query should raise ValueError."""
        with pytest.raises(ValueError, match="Query is too short"):
            rag_chain.get_relevant_documents("Analyze")

    # ── chain initialisation tests ────────────────────────────────

    def test_chain_lazy_initialization(self, rag_chain):
        """Chain should be None until first use."""
        assert rag_chain.chain is None
        rag_chain._get_chain()
        assert rag_chain.chain is not None

    # ── query test — LLM mocked to avoid API cost in CI ──────────

    def test_query_returns_string(self, rag_chain):
        """Query should return a non-empty string."""
        result = rag_chain.query(
            "What is the ECB policy on euro safe assets?",
            reports=[],
            style="balanced",
        )
        assert isinstance(result, str)
        assert len(result) > 0


    def test_query_all_styles(self, rag_chain):
        """All three response styles should return a string."""
        for style in ["balanced", "technical", "non-technical"]:
            result = rag_chain.query(
                "What is the ECB stance on the euro exchange rate?",
                reports=[],
                style=style,
            )
            assert isinstance(result, str)
            assert len(result) > 0
