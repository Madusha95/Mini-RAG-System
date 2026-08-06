import logging
from typing import List
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class RAGResponse(BaseModel):
    """Schema for the final structured response."""
    answer: str = Field(description="Natural language answer to the query based on context.")
    contexts: List[str] = Field(description="Retrieved plot snippets used to form the answer.")
    reasoning: str = Field(description="Short explanation of how the answer was formed.")


class RAGPipeline:
    """Orchestrates document retrieval and Azure LLM output generation."""
    def __init__(self, vector_store_manager: VectorStoreManager, azure_deployment: str, azure_endpoint: str, api_version: str):
        self.vector_store_manager = vector_store_manager
        
        # Initialize Azure Chat OpenAI
        self.llm = AzureChatOpenAI(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            temperature=0.2
        )
        self.parser = JsonOutputParser(pydantic_object=RAGResponse)
        
        self.prompt_template = ChatPromptTemplate.from_template(
            template="""You are a helpful movie plot assistant. Answer the user's question accurately using ONLY the provided contexts below.

Contexts:
{context}

Question: {query}

{format_instructions}
""",
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def query(self, query_str: str, top_k: int = 3) -> RAGResponse:
        """Executes full retrieval and generation workflow."""
        logger.info(f"Processing query: '{query_str}'")
        
        retriever = self.vector_store_manager.get_retriever(top_k=top_k)
        retrieved_docs = retriever.invoke(query_str)
        
        context_snippets = [doc.page_content for doc in retrieved_docs]
        context_str = "\n\n---\n\n".join(context_snippets)

        chain = self.prompt_template | self.llm | self.parser
        raw_response = chain.invoke({"context": context_str, "query": query_str})

        return RAGResponse(
            answer=raw_response.get("answer", ""),
            contexts=context_snippets,
            reasoning=raw_response.get("reasoning", "")
        )