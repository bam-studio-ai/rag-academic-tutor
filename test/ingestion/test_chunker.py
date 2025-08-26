import pytest
from src.ingestion.chunker import TextChunker, TextChunk
from src.ingestion.document import Document


class TestTextChunker:
    @pytest.fixture
    def chunker(self):
        return TextChunker(chunk_size=100, overlap=20, min_chunk_size=30)

    @pytest.fixture
    def sample_document(self):
        content = ("This is a test document. It contains multiple sentences. "
                   "The purpose is to test the chunking functionality. "
                   "We will see how well it handles different chunking methods. "
                   "This should be sufficient for testing.\n\n")
        metadata = {"author": "Test"}
        return Document(content=content, metadata=metadata, src="test.txt")

    def test_sliding_window_chunk(self, chunker, sample_document):
        chunks = chunker.chunk_document(sample_document, chunk_method="sliding_window")
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert len(chunk.content) <= chunker.chunk_size + 1  # +1 for possible trailing space

    def test_paragraph_boundary_chunk(self, chunker, sample_document):
        chunks = chunker.chunk_document(sample_document, chunk_method="paragraph_boundary")
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)

    def test_semantic_chunking(self, chunker, sample_document):
        chunks = chunker.chunk_document(sample_document, chunk_method="semantic")
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert len(chunk.content) <= chunker.chunk_size + 1

    def test_unsupported_chunking_method(self, chunker, sample_document):
        with pytest.raises(ValueError, match="Unsupported chunking method"):
            chunker.chunk_document(sample_document, chunk_method="unknown_method")
