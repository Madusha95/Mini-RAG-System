# 🎬 Mini RAG System (Movie Plots)

A lightweight, production-grade Retrieval-Augmented Generation (RAG) system built to answer questions about Wikipedia movie plots using **LangChain**, **Azure OpenAI**, **ChromaDB**, and **Pydantic**.

This implementation features a clean, Object-Oriented design that enforces strict structured JSON outputs (`answer`, `contexts`, `reasoning`).

---

## 🛠️ Tech Stack & Architecture

- **LLM Engine:** Azure Chat OpenAI (`gpt-5.4` / `gpt-4o-mini`)
- **Embedding Model:** Azure OpenAI Embeddings (`text-embedding-3-small`)
- **Vector Database:** ChromaDB (Local Persistent Storage)
- **Framework:** LangChain (`langchain-openai`, `langchain-chroma`, `langchain-core`)
- **Output Parsing:** Pydantic / LangChain `JsonOutputParser`
- **Data Source:** Wikipedia Movie Plots Dataset (`wiki_movie_plots_deduped.csv`)

---

## 📂 Repository Structure

```text
mini_rag/
├── .env                  # Local API keys and Azure configuration
├── .env.example          # Environment variable template for reviewers
├── .gitignore            # Ignores secrets, cache, and local DB folder
├── config.py             # Central dataclass configuration & validation
├── data_loader.py        # CSV loading and Document object parsing
├── text_chunker.py       # Text splitting logic using RecursiveCharacterTextSplitter
├── vector_store.py       # ChromaDB vector store manager (Persistent/In-Memory)
├── pipeline.py           # RAG chain orchestrator & Pydantic response schema
├── main.py               # Pipeline entry point
├── requirements.txt      # Project dependencies
└── README.md             # Documentation
