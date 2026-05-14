import pytest

from src.rag.document_loader import ECBSpeechLoader
from src.rag.chain import FinancialRAGChain
from src.rag.vector_store import FinancialVectorStore

@pytest.mark.integration
class TestFinancialRAGIntegration:

    def test_full_pipeline(self, test_environment):
        """Test complete RAG pipeline end-to-end with ECB speeches."""

        # initialise components
        loader    = ECBSpeechLoader(test_environment["reports_file"])
        documents = loader.create_documents()

        vector_store = FinancialVectorStore()
        vector_store.create_vector_store(documents)

        rag_chain = FinancialRAGChain(vector_store)

        # test document retrieval — query matches speech content
        docs = rag_chain.get_relevant_documents("euro safe assets financial markets")
        assert len(docs) > 0
        assert "content"  in docs[0]
        assert "metadata" in docs[0]
        assert docs[0]["metadata"]["source"] == "ECB Speeches Dataset"

        response = rag_chain.query(
            "What is the ECB stance on euro safe assets?",
            reports=[],
            style="balanced",
        )
        assert isinstance(response, str)
        assert len(response) > 0

        assert isinstance(response, str)
        assert len(response) > 0

    def test_pipeline_with_different_styles(self, test_environment):
        """Test pipeline produces responses for all audience styles."""

        loader    = ECBSpeechLoader(test_environment["reports_file"])
        documents = loader.create_documents()

        vector_store = FinancialVectorStore()
        vector_store.create_vector_store(documents)

        rag_chain = FinancialRAGChain(vector_store)


        for style in ["balanced", "technical", "non-technical"]:

            response = rag_chain.query(
            "What has the ECB said about digital euro payments?",
            reports=[],
            style=style,
            )
            assert isinstance(response, str)
            assert len(response) > 0