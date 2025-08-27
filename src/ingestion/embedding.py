import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingModel:
    """ 
    A class to handle text embeddings and similarity calculations using a pre-trained model.
    """

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, text: str) -> np.ndarray:
        return self.model.encode(text, 
                                 normalize_embeddings=True,
                                 convert_to_numpy=True)

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        # Cosine similarity
        return float(np.dot(emb1, emb2)) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

