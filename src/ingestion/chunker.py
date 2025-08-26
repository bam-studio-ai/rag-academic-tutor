import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Any
from .document import Document

@dataclass
class TextChunk:
    content: str
    metadata: Dict[str, Any]

class TextChunker:
    def __init__(self, 
                 chunk_size: int = 500, 
                 overlap: int = 50,
                 min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

        self.sentence_endings = re.compile(r'(?<=[.!?]) +')
        self.paragraph_breaks = re.compile(r'\n+')


    def chunk_document(self, document: Document, chunk_method: str = "sliding_window") -> List[TextChunk]:
        if chunk_method == "sliding_window":
            return self._sliding_window_chunk(document)
        elif chunk_method == "semantic":
            return self._semantic_chunking(document)
        elif chunk_method == "paragraph_boundary":
            return self._paragraph_boundary_chunk(document)
        else:
            raise ValueError(f"Unsupported chunking method: {chunk_method}")

    def _sliding_window_chunk(self, document: Document) -> List[TextChunk]:
        content = document.content
        chunks = []

        # Split into sentences for boundary-aware chunking
        sentences = self.sentence_endings.split(content)
        current_chunk = ""
        current_start = 0
        chunk_counter = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += (sentence + " ")
            else:
                if len(current_chunk) >= self.min_chunk_size:
                    chunk_metadata = {
                        **document.metadata,
                        'chunk_index': chunk_counter,
                        'chunk_start': current_start,
                        'chunk_end': current_start + len(current_chunk)
                    }
                    chunks.append(TextChunk(content=current_chunk.strip(), metadata=chunk_metadata))
                    chunk_counter += 1
                    current_start += len(current_chunk) - self.overlap

                current_chunk = sentence + " "

        return chunks

    def _paragraph_boundary_chunk(self, document: Document) -> List[TextChunk]:
        content = document.content
        chunks = []

        paragraphs = self.paragraph_breaks.split(content)
        current_chunk = ""
        current_start = 0
        chunk_counter = 0

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 1 <= self.chunk_size:
                current_chunk += (paragraph + "\n")
            else:
                if len(current_chunk) >= self.min_chunk_size:
                    chunk_metadata = {
                        **document.metadata,
                        'chunk_index': chunk_counter,
                        'chunk_start': current_start,
                        'chunk_end': current_start + len(current_chunk)
                    }
                    chunks.append(TextChunk(content=current_chunk.strip(), metadata=chunk_metadata))
                    chunk_counter += 1
                    current_start += len(current_chunk) - self.overlap

                current_chunk = paragraph + "\n"

        return chunks

    def _semantic_chunking(self, document: Document) -> List[TextChunk]:
        content = document.content
        chunks = []

        sentences = self.sentence_endings.split(content)
        current_chunk = ""
        current_start = 0
        chunk_counter = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += (sentence + " ")
            else:
                if len(current_chunk) >= self.min_chunk_size:
                    chunk_metadata = {
                        **document.metadata,
                        'chunk_index': chunk_counter,
                        'chunk_start': current_start,
                        'chunk_end': current_start + len(current_chunk)
                    }
                    chunks.append(TextChunk(content=current_chunk.strip(), metadata=chunk_metadata))
                    chunk_counter += 1
                    current_start += len(current_chunk) - self.overlap

                current_chunk = sentence + " "

        return chunks
