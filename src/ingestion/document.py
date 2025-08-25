import os
import io
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class Document:
    """A simple Document class to hold text content and metadata."""
    content: str
    metadata: Dict[str, Any]
    src: str

class DocumentLoader:
    """
    A loader to read text files and create Document objects.
    """
    def __init__(self):
        self.supported_extensions = ['.txt']

    def load_document(self, file_path: str) -> Document:
        """Load a single document from a file path."""
        _, ext = os.path.splitext(file_path)
        if ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {ext}")

        with io.open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata = {
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'file_extension': ext,
            'file_path': file_path
        }

        return Document(content=content.strip(), 
                        metadata=metadata, 
                        src=file_path)

    def load_documents(self, file_paths: List[str]) -> [Document]:
        """Load multiple documents from a list of file paths."""
        documents = []
        for file_path in file_paths:
            documents.append(self.load_document(file_path))
        return documents

    def load_directory(self, dir_path: str, recursive: bool) -> [Document]:
        """Load all supported documents from a directory."""
        file_paths = []

        for root, _, files in os.walk(dir_path):
            for file in files:
                if any(file.endswith(ext) for ext in self.supported_extensions):
                    file_paths.append(os.path.join(root, file))
            if not recursive:
                break

        return self.load_documents(file_paths)
