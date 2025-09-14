from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.ingestion.embedding import EmbeddingModel
from .prompt_template import PromptTemplate
from .llm_client import LLMClient

from dataclasses import dataclass
from typing import List

@dataclass
class RAGResponse:
    answer: str
    source_documents: List


class RAGChain:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
                 persist_directory: str = "./data/chroma_db",
                 collection_name: str = "academic_docs"):
        self.vector_store = VectorStore(persist_directory=persist_directory,
                                        collection_name=collection_name, 
                                        embedding_model=EmbeddingModel())

        self.llm_client = LLMClient(model_name=model_name)

        self.hybrid_retriever = HybridRetriever(vector_store=self.vector_store)

        self._index_for_hybrid_search()


    def _index_for_hybrid_search(self):
        """
        Index documents in the vector store for hybrid search.
        """
        all_docs = self.vector_store.get_all_documents()
        print(f"Indexing {len(all_docs)} documents for hybrid search.")
        contents = [doc['content'] for doc in all_docs]
        doc_ids = [doc['id'] for doc in all_docs]
        self.hybrid_retriever.index_documents(contents, doc_ids)

    def ask_hybrid(self, question: str) -> RAGResponse:
        """Process a question and return a response using the hybrid RAG approach."""
        # Step 1: Retrieve relevand docs using hybrid retriever
        search_response = self.hybrid_retriever.search(question, top_k=3)
        relevant_docs = [{'content': doc.content} for doc in search_response]

        # Step 2: Generate prompts using the docs
        prompt = PromptTemplate.generate_prompt(question, relevant_docs)

        # Step 3: Get response from LLM
        llm_response = self.llm_client.get_response(prompt)

        return RAGResponse(answer=llm_response,
                           source_documents=search_response)



    def ask(self, question: str) -> RAGResponse:
        """Process a question and return a response using the RAG approach."""
        # Step 1: Retrieve relevant documents 
        search_response = self.vector_store.search(question, top_k=3)
        relevant_docs = search_response['results']

        # Step 2: Generate prompt
        prompt = PromptTemplate.generate_prompt(question, relevant_docs)

        # Step 3: Get response from LLM
        llm_response = self.llm_client.get_response(prompt)

        return RAGResponse(answer=llm_response,
                           source_documents=relevant_docs)

