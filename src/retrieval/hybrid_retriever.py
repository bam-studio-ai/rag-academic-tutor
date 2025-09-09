from .vector_store import VectorStore
from rank_bm25 import BM25Okapi
from typing import List
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    id: str
    content: str
    score: float

class HybridRetriever:
    def __init__(self, vector_store: VectorStore, alpha: int = 0.5):
        """
        Initialize the HybridRetriever with a vector store and alpha parameter.

        :param vector_store: An instance of VectorStore for vector-based retrieval.
        :param alpha: Weighting factor between vector and keyword search (0 <= alpha <= 1).
        """
        self.vector_store = vector_store
        self.alpha = alpha
        self.documents = []
        self.doc_ids = []
        self.bm25 = None

    def index_documents(self, documents: List[str], doc_ids: List[str]):
        """
        Index documents for BM25 keyword search.

        :param documents: List of document texts to index.
        :param doc_ids: List of document IDs corresponding to the texts.
        """
        self.documents = documents
        self.doc_ids = doc_ids

        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Perform a hybrid search using both vector and keyword-based retrieval.

        :param query: The search query string.
        :param top_k: Number of top results to return.
        :return: List of RetrievalResult objects containing the top_k results.
        """

        # Vector search
        vector_results = self.vector_store.search(query, top_k)

        # BM25 Search
        bm25_scores = self.bm25.get_scores(query.lower().split())

        # Combine results
        hybrid_scores = {}

        for result in vector_results['results']:
            doc_id = result['id']
            vector_score = result.get('score', 0)

            hybrid_scores[doc_id] = {
                'vector_score': vector_score,
                'bm25_score': 0,
                'content': result.get('content', ''),
            }

        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1

        for i, score in enumerate(bm25_scores):
            doc_ids = self.doc_ids[i]
            normalized_score = score / max_bm25

            if doc_id in hybrid_scores:
                hybrid_scores[doc_id]['bm25_score'] = normalized_score
            else:
                hybrid_scores[doc_id] = {
                    'vector_score': 0,
                    'bm25_score': normalized_score,
                    'content': self.documents[i],
                }

        # Calculate final scores and prepare results
        results = []
        for doc_id, scores in hybrid_scores.items():
            final_score = (
                self.alpha * scores['vector_score'] +
                (1 - self.alpha) * scores['bm25_score']
            )

            results.append(RetrievalResult(
                id=doc_id,
                content=scores['content'],
                score=final_score
            ))

        # Sort and return top_k results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
