import pytest
from src.generation.rag_chain import RAGChain

class TestRAGChain:
    @pytest.fixture
    def rag_chain(self):
        return RAGChain()

    def test_initialization(self, rag_chain):
        assert rag_chain is not None
        assert hasattr(rag_chain, 'vector_store')

    def test_ask_method(self, rag_chain):
        question = "What is the capital of France?"
        response = rag_chain.ask(question)
        assert response is not None
        assert hasattr(response, 'answer')
        assert hasattr(response, 'source_documents')
        assert isinstance(response.source_documents, list)
        # Let's assume the placeholder answer is returned
