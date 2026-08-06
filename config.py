import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class RAGConfig:
    """Central configuration for Azure OpenAI models, chunking, and paths."""
    dataset_path: str = "wiki_movie_plots_deduped.csv"
    sample_size: int = 300
    chunk_size: int = 1200
    chunk_overlap: int = 150
    chroma_collection_name: str = "movie_plots"
    persist_directory: str = "./chroma_db"  # Local folder to store Chroma DB files
    top_k: int = 3

    # Azure OpenAI Configurations
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    llm_deployment: str = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-5.4")
    embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

    @staticmethod
    def validate_env():
        required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")