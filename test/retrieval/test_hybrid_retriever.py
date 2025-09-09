import pytest
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore
from src.ingestion.embedding import EmbeddingModel

class TestHybridRetriever:
    @pytest.fixture
    def hybrid_retriever(self):
        vector_store = VectorStore(
            persist_directory="./test_persist",
            collection_name="test_collection",
            embedding_model=EmbeddingModel()
        )

        return HybridRetriever(
            vector_store=vector_store 
        )

    def test_initialization(self, hybrid_retriever):
        assert hybrid_retriever.vector_store is not None

    def test_index_documents(self, hybrid_retriever):
        docs = [
            "This is a test document.",
            "This is another test document."
        ]
        doc_ids = ["doc1", "doc2"]
        hybrid_retriever.index_documents(docs, doc_ids)
        assert "This is a test document." in hybrid_retriever.documents
        assert "doc1" in hybrid_retriever.doc_ids

    def test_search(self, hybrid_retriever):
        docs = [
            "This is a test document.",
            "This is another test document.",
            "This document is different.",
            "Completely unrelated content here.",
            "Test document with some common words."
        ]
        doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        hybrid_retriever.index_documents(docs, doc_ids)
        results = hybrid_retriever.search("test document", top_k=3)
        assert len(results) == 3






