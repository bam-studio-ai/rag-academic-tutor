import pytest
from src.ingestion.embedding import EmbeddingModel
import numpy as np

class TestEmbeddingModel:
    @pytest.fixture
    def embedding_model(self):
        return EmbeddingModel()

    def test_encode(self, embedding_model):
        text = "This is a test sentence."
        embedding = embedding_model.encode(text)
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1  # Should be a 1D array

    def test_similarity(self, embedding_model):
        text1 = "This is a test sentence."
        text2 = "This is another test sentence."
        emb1 = embedding_model.encode(text1)
        emb2 = embedding_model.encode(text2)
        sim = embedding_model.similarity(emb1, emb2)
        assert -1.0 <= sim <= 1.0  # Cosine similarity range
