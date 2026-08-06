import os
import logging
from typing import List, Optional
import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class MovieDataLoader:
    """Handles loading and parsing raw movie plot datasets."""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_documents(self, sample_size: Optional[int] = None) -> List[Document]:
        """Loads dataset from CSV and converts rows into LangChain Document objects."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")

        logger.info(f"Loading raw dataset from {self.file_path}...")
        df = pd.read_csv(self.file_path)
        
        df = df.dropna(subset=['Plot', 'Title'])
        if sample_size:
            df = df.head(sample_size)
            print(df.head(5))
            
        documents = []
        for _, row in df.iterrows():
            text_content = f"Movie Title: {row['Title']}\nPlot: {row['Plot']}"
            metadata = {"title": str(row["Title"])}
            documents.append(Document(page_content=text_content, metadata=metadata))

        print(documents[0:10])

        logger.info(f"Successfully loaded {len(documents)} document objects.")
        
        return documents
