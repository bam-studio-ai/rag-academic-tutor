import pytest
from src.retrieval.vector_store import VectorStore
from src.ingestion.chunker import TextChunk
from src.ingestion.embedding import EmbeddingModel

class TestVectorStore:
    @pytest.fixture
    def vector_store(self):
        return VectorStore(
            persist_directory="./test_persist",
            collection_name="test_collection",
            embedding_model=EmbeddingModel()
        )

    def test_initialization(self, vector_store):
        assert vector_store.persist_directory == "./test_persist"
        assert vector_store.collection_name == "test_collection"
        assert vector_store.client is not None
        assert vector_store.collection is not None

    def test_add_documents(self, vector_store):
        chunkd = [
            TextChunk(
                content="This is a test chunk.",
                metadata={"source": "test.txt"},
            ),
            TextChunk(
                content="This is another test chunk.",
                metadata={"source": "test.txt"}
            )
        ]

        result = vector_store.add_documents(chunkd)
        assert result is not None
        assert result["added"] == 2
        assert result["failed"] == 0


