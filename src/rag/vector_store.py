import logging
logger = logging.getLogger(__name__)

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import os


class FinancialVectorStore:
    """ - Creation/save of vector store
        - Similarity search
        - Load vector store
        - Upsert documents (always add fresh, FAISS surfaces most relevant)
    """

    def __init__(self):
        self.vector_store = None
        self.embeds = OpenAIEmbeddings(model="text-embedding-3-small")

    def create_vector_store(self, documents: List[Document]) -> None:
        print(f"DEBUG {len(documents)} documents received")
        for i, doc in enumerate(documents):
            print(f"DEBUG doc {i}: type={type(doc)}, content_len={len(doc.page_content)}")

        embedding_dim = len(self.embeds.embed_query("hello ECB analyst"))
        index = faiss.IndexFlatL2(embedding_dim)

        self.vector_store = FAISS(
            embedding_function=self.embeds,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.vector_store.add_documents(documents)

    def upsert_documents(self, documents: List[Document],
                         save_path: str = None) -> Dict[str, int]:
        """
        Add documents to the vector store, always inserting fresh data.
        FAISS has no native delete — we add all documents and rely on
        similarity search to surface the most relevant (freshest) content.

        Args:
            documents:  List of LangChain Documents
            save_path:  If provided, persist the updated store to disk.

        Returns:
            {"added": N, "skipped": 0}
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialised — call create_vector_store first.")

        if documents:
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store.")

            if save_path:
                self.save_local(save_path)
                logger.info(f"Vector store saved to {save_path}.")

        return {"added": len(documents), "skipped": 0}

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if self.vector_store is None:
            raise ValueError("Vector store not initialised")
        if not query or not query.strip():
            logger.warning("Empty query received, returning empty list")
            return []
        if len(query.strip()) < 10:
            logger.warning("Query too short, returning empty list")
            return []
        return self.vector_store.similarity_search(query=query, k=k)

    def save_local(self, path: str) -> None:
        self.vector_store.save_local(path)

    @classmethod
    def load_local(cls, directory: str) -> "FinancialVectorStore":
        instance = cls()
        instance.vector_store = FAISS.load_local(
            directory,
            instance.embeds,
            allow_dangerous_deserialization=True,
        )
        return instance