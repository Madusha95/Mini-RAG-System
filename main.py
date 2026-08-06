import os
import json
import logging
from config import RAGConfig
from data_loader import MovieDataLoader
from text_chunker import TextChunker
from vector_store import VectorStoreManager
from pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    # 0. Load settings and validate Azure environment variables
    config = RAGConfig()
    config.validate_env()

    # 1. Instantiate Vector Store Manager
    vector_mgr = VectorStoreManager(
        azure_deployment=config.embedding_deployment,
        azure_endpoint=config.azure_endpoint,
        api_version=config.api_version,
        collection_name=config.chroma_collection_name,
        persist_directory=config.persist_directory
    )

    # 2. Check if persistent DB exists; if not, load and chunk data
    db_exists = os.path.exists(config.persist_directory) and len(os.listdir(config.persist_directory)) > 0

    if not db_exists:
        logging.info("Local database not found. Starting data preprocessing and embedding workflow...")
        
        # Load Data
        loader = MovieDataLoader(file_path=config.dataset_path)
        raw_docs = loader.load_documents(sample_size=config.sample_size)

        # Chunk Documents
        chunker = TextChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        chunks = chunker.split_documents(raw_docs)

        # Build & Persist Store
        vector_mgr.initialize_store(chunks=chunks)
    else:
        # Load directly from disk without re-embedding
        vector_mgr.initialize_store()

    # 3. Instantiate & Query RAG Engine
    pipeline = RAGPipeline(
        vector_store_manager=vector_mgr,
        azure_deployment=config.llm_deployment,
        azure_endpoint=config.azure_endpoint,
        api_version=config.api_version
    )
    
    #query = "What happens in the plot of Love by the Light of the Moon?"
    #query = "What action does the bartender take against Carrie Nation in Kansas Saloon Smashers?"
    query = "How does Daniel Boone escape from the stake in the 1907 film Daniel Boone?"
    result = pipeline.query(query, top_k=config.top_k)

    # 4. Output Results
    print("\n" + "="*50)
    print("FINAL STRUCTURED OUTPUT (JSON):")
    print("="*50)
    print(json.dumps(result.model_dump(), indent=2))

if __name__ == "__main__":
    main()