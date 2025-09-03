from src.ingestion.document import DocumentLoader
from src.ingestion.chunker import TextChunker
from src.ingestion.embedding import EmbeddingModel
from src.retrieval.vector_store import VectorStore

# Load documents 
doc_loader = DocumentLoader()
documents = doc_loader.load_directory('./docs/raw', recursive=True)
print(f"Loaded {len(documents)} documents.")

# Chunk documents
chunker = TextChunker(chunk_size=500, overlap=50, min_chunk_size=100)
chunks = [] 

for document in documents:
    chunks += chunker.chunk_document(document=document,
                                         chunk_method="sliding_window")

print(f"Created {len(chunks)} chunks from documents.")

# Store them
embedding_model = EmbeddingModel()
vector_store = VectorStore(persist_directory='./data/chromadb/',
                           collection_name='aiaa_docs',
                           embedding_model=embedding_model)
vector_store._reset_collection()

result = vector_store.add_documents(chunks, batch_size=100)
print(f"Added {result['added']} chunks to the vector store.")
print(f"Failed to add {result['failed']} chunks to the vector store.")
print(f"Total chunks {result['total']} in the vector store.")
print("Ingestion complete.")

