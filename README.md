# Medical RAG System

A Retrieval-Augmented Generation (RAG) pipeline for medical document question answering, built from scratch as a learning project.

## What it does
- Ingests medical text documents and splits them into chunks
- Embeds each chunk using a local sentence transformer model (all-MiniLM-L6-v2)
- Stores embeddings in a ChromaDB vector store for semantic search
- Retrieves the most relevant chunks for any question
- Passes retrieved context to a local LLM (Llama 3.2 via Ollama) to generate grounded answers

## Why RAG?
Standard LLMs answer from general training data and can hallucinate. This system grounds every answer in specific retrieved documents, making responses traceable and verifiable — critical for medical applications.

## Project structure
medical-rag/

├── data/          ← medical text documents

├── db/            ← ChromaDB vector store (auto-generated)

├── ingest.py      ← loads, chunks, embeds, and stores documents

├── retrieve.py    ← semantic search over stored chunks

└── query.py       ← full pipeline: retrieve + LLM answer

## Setup

1. Clone the repo
2. Create and activate a conda environment:
conda create -n medical-rag python=3.12

conda activate medical-rag
3. Install dependencies:
pip install anthropic chromadb sentence-transformers rich google-genai python-dotenv
4. Install Ollama from ollama.com and pull the model:
ollama pull llama3.2
5. Run ingestion to build the vector store: python ingest.py
6. Ask questions:python query.py
## Tech stack
- **ChromaDB** — vector store for semantic search
- **sentence-transformers** — local embedding model
- **Ollama + Llama 3.2** — local LLM, no API key required
- **Python 3.12**

## What I learned
- How chunking strategy directly affects retrieval quality
- How embeddings represent meaning as vectors in high-dimensional space
- Why grounding LLM responses in retrieved context prevents hallucination
- The tradeoffs between chunk size, overlap, and retrieval precision