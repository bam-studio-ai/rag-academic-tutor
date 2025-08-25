import pytest
import os
import tempfile
from src.ingestion.document import Document, DocumentLoader


class TestDocument:
    def test_document_creation(self):
        doc = Document(content="Hello", 
                       metadata={"author": "Alice"}, 
                       src="file.txt")
        assert doc.content == "Hello"
        assert doc.metadata == {"author": "Alice"}
        assert doc.src == "file.txt"

class TestDocumentLoader:
    @pytest.fixture
    def loader(self):
        return DocumentLoader()

    @pytest.fixture
    def temp_txt_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content.")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_load_document_success(self, loader, temp_txt_file):
        doc = loader.load_document(temp_txt_file)
        assert doc.content == "Test file content."
        assert doc.src == temp_txt_file
        assert doc.metadata['file_extension'] == '.txt'

    def test_load_document_unsupported_extension(self, loader):
        with pytest.raises(ValueError, match="Unsupported file extension"):
            loader.load_document("file.pdf")

    def test_load_documents(self, loader):
        # Create multiple temp files
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"Content of file {i}.")
                temp_files.append(f.name)

        try:
            docs = loader.load_documents(temp_files)
            assert len(docs) == 3
            for i, doc in enumerate(docs):
                assert doc.content == f"Content of file {i}."

        finally:
            for file in temp_files:
                os.unlink(file)

    def test_load_directory(self, loader):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files in the directory
            file_paths = []
            for i in range(2):
                file_path = os.path.join(temp_dir, f"file{i}.txt")
                with open(file_path, 'w') as f:
                    f.write(f"Content of file {i}.")
                file_paths.append(file_path)

            docs = loader.load_directory(temp_dir, recursive=False)
            assert len(docs) == 2
