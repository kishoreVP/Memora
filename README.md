<img src="./assets//memora_banner.png" alt="memora banner">

A terminal-based personal memory assistant. Ingest local files, search semantically, and chat with your documents using RAG. Uses OpenRouter for LLM access (free tier available).

<!-- ## Quick Install -->
<!-- 
```bash
bash scripts/install.sh
``` -->

## Prerequisites

- **Python 3.10+**
- **OpenRouter API key** — free at [openrouter.ai/keys](https://openrouter.ai/keys)

## Setup

```bash
uv sync

cp .env.example .env
```

## Usage

```bash

uv run memora add document.pdf
uv run memora add ./notes -r

uv run memora ask "What are the key findings?"

uv run memora chat

uv run memora list
uv run memora stats
uv run memora remove /path/to/file.txt
```

## Architecture

- **LLM**: OpenRouter (Gemini Flash, GPT-4o-mini, Llama, etc.)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, local)
- **Vector Store**: FAISS (cosine similarity)
- **Keyword Search**: BM25
- **Retrieval**: Hybrid FAISS + BM25 with Reciprocal Rank Fusion
- **Orchestration**: LangChain
- **TUI**: Textual
- **CLI**: Typer
- **Storage**: JSON (metadata) + FAISS index (vectors)