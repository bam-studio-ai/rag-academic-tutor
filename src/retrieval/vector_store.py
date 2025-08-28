import chromadb
from chromadb.config import Settings
import logging
from pathlib import Path
from typing import Dict, Any
import uuid

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, 
                 persist_directory: str = None, 
                 collection_name: str = "academic_docs",
                 embedding_model=None):
        """Initialize the VectorStore with ChromaDB."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Ensure the persistence directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize client and collection
        self._init_client()
        self._init_collection()

        logger.info(f"VectorStore initialized with collection: {self.collection_name}")


    def _init_client(self):
        """Initialize the Chroma client."""
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"Chroma client initialized with persistence at: {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma client: {e}")
            raise RuntimeError(f"Chroma client initialization error: {e}")


    def _init_collection(self):
        try:
            # Try to get the existing collection
            try: 
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Using existing collection: {self.collection_name}")
            except Exception:
                # If it doesn't exist, create a new one
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "description": "Collection for academic documents",
                        "created_by": "RAG Academic System"
                    }
                )
                logger.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
            # If any other error occurs let's log and raise
            logger.error(f"Failed to initialize or create collection: {e}")
            raise RuntimeError(f"Collection initialization error: {e}")

    def add_documents(self, chunks: list, batch_size: int = 100) -> Dict[str, Any]:
        """Add documents to the vector store in batches."""
        if not chunks:
            logger.warning("No chunks provided to add to the vector store.")
            return {"added": 0, "failed": 0}

        total_added = 0
        total_failed = 0

        ids = []
        metadatas = []
        documents = []
        embeddings_list = []

        # Prepare data for insertion
        for i, chunk in enumerate(chunks):
            chunk_id = f"{uuid.uuid4()}"

            ## Append the id for this chunk
            ids.append(chunk_id)

            # Append the document content for this chunk
            documents.append(chunk.content)
            metadata = chunk.metadata.copy() 

            metadata.update({
                "chunk_id": chunk_id,
                "content_length": len(chunk.content)
            })

            # Append the metadata for this chunk
            metadatas.append(metadata)

            embeddings = self.embedding_model.encode(chunk.content)

            embeddings_list.append(embeddings)

        # Insert in batches
        for i in range(0, len(chunks), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_embeddings = embeddings_list[i:i + batch_size]

            try:
                self.collection.add(
                    ids=batch_ids,
                    metadatas=batch_metadatas,
                    documents=batch_documents,
                    embeddings=batch_embeddings
                )
                total_added += len(batch_ids)
                logger.info(f"Added batch of {len(batch_ids)} documents to the vector store.")
            except Exception as e:
                total_failed += len(batch_ids)
                logger.error(f"Failed to add batch of documents: {e}")

        # Log result
        logger.info(f"Finished adding documents. Total added: {total_added}, Total failed: {total_failed}")

        # Return summary
        return {
            "added": total_added,
            "failed": total_failed, 
            "total": self.collection.count()
        }








