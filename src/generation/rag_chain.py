from src.retrieval.vector_store import VectorStore
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
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.vector_store = VectorStore(persist_directory="./data/chroma_db",
                                        collection_name="academic_docs", 
                                        embedding_model=EmbeddingModel())

        self.llm_client = LLMClient(model_name=model_name)


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



