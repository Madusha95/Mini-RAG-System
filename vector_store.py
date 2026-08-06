import os
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages creation, loading, and querying of a persistent local Chroma vector database."""
    def __init__(
        self, 
        azure_deployment: str, 
        azure_endpoint: str, 
        api_version: str, 
        collection_name: str = "movie_plots",
        persist_directory: str = "./chroma_db"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize Azure OpenAI Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )
        self.vector_store: Optional[Chroma] = None

    def initialize_store(self, chunks: Optional[List[Document]] = None) -> None:
        """
        Loads an existing persistent Chroma store from disk, 
        or creates and persists a new one if it does not exist.
        """
        # Check if persistent DB directory exists and is non-empty
        db_exists = os.path.exists(self.persist_directory) and len(os.listdir(self.persist_directory)) > 0

        if db_exists:
            logger.info(f"Loading existing persistent Chroma database from '{self.persist_directory}'...")
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info("Persistent vector store loaded successfully.")
        else:
            if not chunks:
                raise ValueError("No chunks provided to build a new vector store.")

            logger.info(f"Creating new Chroma database and persisting to '{self.persist_directory}'...")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
            logger.info("Vector store created and saved locally.")

    def get_retriever(self, top_k: int = 3):
        """Returns retriever interface for similarity search."""
        if not self.vector_store:
            raise RuntimeError("Vector store is not initialized. Run initialize_store() first.")
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})