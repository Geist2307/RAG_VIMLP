from pathlib import Path
import pytest
from src.document_loader import FinancialDocumentLoader
from src.rag_chain import FinancialRAGChain
from src.vector_store import FinancialVectorStore


class TestFinancialRAGIntegration:

    def test_full_pipeline(self, test_environment):
        """Test complete RAG pipeline with ECB FX data"""

        # initialize components
        loader = FinancialDocumentLoader(test_environment["reports_file"])
        documents = loader.create_documents()

        vector_store = FinancialVectorStore()
        vector_store.create_vector_store(documents)

        rag_chain = FinancialRAGChain(vector_store)

        # test document retrieval
        fx_docs = rag_chain.get_relevant_documents("US dollar Euro exchange rate")
        assert len(fx_docs) > 0
        assert "ECB-FX-001" in fx_docs[0]["metadata"]["report_id"] # should be equal
        assert "US dollar" in fx_docs[0]["content"] or "USD" in fx_docs[0]["content"]

        # test USD/EUR query
        usd_response = rag_chain.query("What is the USD EUR exchange rate trend in 2025?")
        assert isinstance(usd_response, str)
        assert len(usd_response) > 0
        assert any(term in usd_response for term in ["2025", "USD", "dollar", "Euro"])

        # test GBP/EUR query
        gbp_response = rag_chain.query("What is the pound sterling Euro exchange rate trend?")
        assert isinstance(gbp_response, str)
        assert len(gbp_response) > 0
        assert any(term in gbp_response for term in ["GBP", "pound", "sterling", "Euro"])