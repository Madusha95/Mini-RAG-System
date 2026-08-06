# 🎬 Mini RAG System (Movie Plots)

A lightweight, production-grade Retrieval-Augmented Generation (RAG) system built to answer questions about Wikipedia movie plots using **LangChain**, **Azure OpenAI**, **ChromaDB**, and **Pydantic**.

This implementation features a clean, object-oriented design that enforces strict structured JSON outputs (`answer`, `contexts`, and `reasoning`).

---

# 🛠️ Tech Stack & Architecture

* **LLM Engine:** Azure Chat OpenAI (`gpt-5.4` / `gpt-4o-mini`)
* **Embedding Model:** Azure OpenAI Embeddings (`text-embedding-3-small`)
* **Vector Database:** ChromaDB (Persistent Local Storage)
* **Framework:** LangChain (`langchain-openai`, `langchain-chroma`, `langchain-core`)
* **Output Parsing:** Pydantic + LangChain `JsonOutputParser`
* **Data Source:** Wikipedia Movie Plots Dataset (`wiki_movie_plots_deduped.csv`)

---

# 📂 Repository Structure

```text
mini_rag/
├── .env                      # Local API keys and Azure configuration
├── .env.example              # Environment variable template
├── .gitignore                # Ignores secrets, cache, and local DB folder
├── config.py                 # Central configuration & validation
├── data_loader.py            # CSV loading and Document parsing
├── text_chunker.py           # Text splitting using RecursiveCharacterTextSplitter
├── vector_store.py           # ChromaDB vector store manager
├── pipeline.py               # RAG pipeline and response schema
├── main.py                   # Application entry point
├── requirements.txt          # Project dependencies
└── README.md                 # Documentation
```

---

# 🚀 Getting Started

## 1. Prerequisites

* Python **3.10+**
* An active **Azure OpenAI** deployment (or OpenAI API key)
* `wiki_movie_plots_deduped.csv` placed in the project root

---

## 2. Installation

Clone the repository, create a virtual environment, and install the dependencies.

```bash
# Clone the repository
git clone https://github.com/<your-username>/mini-rag-movie-plots.git

# Navigate to the project
cd mini-rag-movie-plots

# Create a virtual environment
python -m venv venv

# Activate the virtual environment

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 3. Environment Setup

Copy `.env.example` to `.env`.

```bash
cp .env.example .env
```

Then configure your Azure OpenAI credentials.

```env
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-xx-xx
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

---

# 🧪 Sample Output

The pipeline returns a strictly structured JSON response containing the generated answer, retrieved contexts, and reasoning.

```json
{
  "answer": "Daniel Boone is rescued from the stake by his horse after a burning arrow sets the Indian camp on fire, causing panic.",
  "contexts": [
    "Movie Title: Daniel Boone\nPlot: Boone's daughter befriends an Indian maiden as Boone and his companion start out on a hunting expedition... A burning arrow gets shot into the Indian camp. Boone gets tied to the stake and tortured. The burning arrow sets the Indian camp on fire, causing panic. Boone is rescued by his horse, and Boone has a knife fight in which he kills the Indian chief."
  ],
  "reasoning": "The retrieved context explicitly states that Boone is tied to a stake when a burning arrow ignites the Indian camp, causing panic and allowing his horse to rescue him."
}
```

---

# 🎥 Video Walkthrough

A short walkthrough explaining the project architecture, implementation, and execution.

👉 **Watch the Loom Video Here**

```
https://www.loom.com/share/your-loom-link-here
```

---

# 🗄️ ChromaDB Integration

This project uses **ChromaDB** as the vector database to store and retrieve embeddings generated from Wikipedia movie plot descriptions.

## Features

### ✅ Persistent Local Storage

The vector database is configured with:

```python
persist_directory="./chroma_db"
```

Embeddings are stored on disk, allowing the application to reload them instantly on subsequent runs without regenerating embeddings. This significantly reduces API costs and startup time.

### ✅ Efficient Similarity Search

* Retrieves the **Top-K (k=3)** most relevant movie plot chunks.
* Uses vector similarity (Cosine/Euclidean distance depending on configuration).
* Provides relevant context for the language model to generate accurate answers.

### ✅ Seamless LangChain Integration

The vector store is managed using `langchain_chroma.Chroma`, providing a clean abstraction for:

* Document ingestion
* Embedding storage
* Similarity search
* Retriever creation
* Persistent database management

---

# 📌 Response Schema

The application returns responses in a strict Pydantic schema.

```json
{
  "answer": "string",
  "contexts": [
    "string"
  ],
  "reasoning": "string"
}
```

This guarantees consistent, machine-readable outputs that are easy to integrate with downstream applications or APIs.
