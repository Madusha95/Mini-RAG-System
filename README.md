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

## 🚀 Getting Started

### 1. Prerequisites
* **Python:** Version 3.10 or higher
* **API Key:** An active Azure OpenAI deployment (or standard OpenAI API key)
* **Dataset:** `wiki_movie_plots_deduped.csv` placed in the root directory

### 2. Installation
Clone the repository, create a virtual environment, and install dependencies:

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/mini-rag-movie-plots.git
cd mini-rag-movie-plots

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

### 3. Environment Setup
Copy `.env.example` to `.env` and fill in your Azure OpenAI parameters:

```bash
cp .env.example .env

## 🧪 Sample Output

The pipeline returns a strictly structured JSON object containing the answer, retrieved context snippets, and the reasoning process:

```json
{
  "answer": "Daniel Boone is rescued from the stake by his horse after a burning arrow sets the Indian camp on fire, causing panic.",
  "contexts": [
    "Movie Title: Daniel Boone\nPlot: Boone's daughter befriends an Indian maiden as Boone and his companion start out on a hunting expedition... A burning arrow gets shot into the Indian camp. Boone gets tied to the stake and tortured. The burning arrow sets the Indian camp on fire, causing panic. Boone is rescued by his horse, and Boone has a knife fight in which he kills the Indian chief.[2]"
  ],
  "reasoning": "The retrieved plot context for 'Daniel Boone' explicitly states that while Boone is tied to a stake and tortured, a burning arrow sets the Indian camp on fire, causing panic, and Boone is rescued by his horse."
}

## 🎥 Video Walkthrough

Check out the 2-minute Loom walkthrough explaining the architecture, design choices, and demonstrating execution:

👉 **[Watch the Loom Video Walkthrough Here](https://www.loom.com/share/your-loom-link-here)**

## 🗄️ Chroma Vector Database Integration

This project uses **ChromaDB** as its vector database to store and query high-dimensional embeddings generated from movie plot descriptions.

### Key Highlights & Capabilities:
* **Persistent Local Storage:** Configured with `persist_directory="./chroma_db"`, ensuring embeddings are saved directly to disk. On subsequent application runs, the system reloads existing embeddings instantly, eliminating unnecessary API embedding calls and drastically reducing latency.
* **Efficient Similarity Search:** Leverages Cosine/Euclidean distance indexing to perform fast vector retrieval ($k=3$), fetching only the most contextually relevant movie plot chunks for the LLM.
* **LangChain Integration:** Seamlessly managed via `langchain_chroma.Chroma`, enabling clean abstraction between vector indexing, data ingestion, and retrieval chains.

