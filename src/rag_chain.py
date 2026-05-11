
"""
src/rag_chain.py
----------------
RAG chain for ECB FX analysis.
Context is assembled from:
  1. Retrieved vector store documents
  2. Live report data (model info, predictions, forecast horizon)
"""

import logging
logger = logging.getLogger(__name__)

from typing import Dict, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from operator import itemgetter

from .vector_store import FinancialVectorStore
from .trend_enricher import format_fact_context


def _docs_to_str(docs) -> str:
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'ECB')} | "
        f"Period: {doc.metadata.get('publication_date', 'N/A')}]\n{doc.page_content}"
        for doc in docs
    )


def _merge_contexts(doc_str: str, fact_str: str) -> str:
    parts = []
    if doc_str.strip():
        parts.append("── RETRIEVED DOCUMENTS ──\n" + doc_str)
    if fact_str.strip():
        parts.append("── LIVE MODEL & MARKET DATA ──\n" + fact_str)
    return "\n\n".join(parts)


class FinancialRAGChain:
    """
    RAG chain for ECB FX analysis.

    Query validation:
      - Empty query → ValueError
      - Fewer than 10 characters → ValueError

    Context is hybrid:
      - Vector store documents (retrieved)
      - Live reports from query_agent (model info + predictions)
    """

    def __init__(self, vector_store: FinancialVectorStore):
        self.vector_store = vector_store
        self.llm          = ChatOpenAI(model="gpt-5.5", temperature=0)
        self.chain        = None

    def _get_chain(self):
        if self.chain is None:
            self.chain = self._create_chain()
        return self.chain

    def _validate_query(self, query: str) -> None:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty. Please provide an input")
        if len(query.strip()) < 10:
            raise ValueError("Query is too short. Please provide more context")

    def _create_chain(self):
        retriever = self.vector_store.vector_store.as_retriever()

        def build_context(inputs: dict) -> str:
            question = inputs["question"]
            reports  = inputs.get("reports", [])
            docs     = retriever.invoke(question)
            doc_str  = _docs_to_str(docs)
            fact_str = format_fact_context(reports)   # live reports, not JSON
            return _merge_contexts(doc_str, fact_str)

        prompt = ChatPromptTemplate.from_template("""
You are a financial analyst assistant specialising in ECB foreign exchange data.
A Bayesian Variational Dropout MLP (Molchanov et al. 2017) has been applied to \
the data and the posterior predictive chart is shown to the user separately.

RULES:
- Always cite the latest observed value, currency pair, and date explicitly
- Describe the Bayesian model: architecture, training schedule, ELBO loss, training date
- State the forecast horizon and number of posterior samples used
- Tell the user the chart shows the posterior predictive mean and uncertainty ribbon (±1σ)
- DO NOT speculate about future prices — refer the user to the chart for the forecast
- DO NOT say you lack trend information — the chart handles this visually
- If asked about trend direction, tell the user to refer to the chart
- Cite source as ECB Data API

CONTEXT: {context}

QUESTION: {question}

ANSWER:
""")

        chain = (
            {
                "context":  RunnableLambda(build_context),
                "question": itemgetter("question"),
                "reports":  itemgetter("reports"),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def query(self, question: str, reports: list = None) -> str:
        self._validate_query(question)
        chain = self._get_chain()
        return chain.invoke({
            "question": question,
            "reports":  reports or [],
        })

    def get_relevant_documents(self, query: str, k: int = 3) -> List[Dict]:
        self._validate_query(query)
        docs = self.vector_store.similarity_search(query=query, k=k)
        return [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]