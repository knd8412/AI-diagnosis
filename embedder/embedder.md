# Embedder (Mistral + Pinecone)

An AI-powered medical assistant that performs semantic search over a medical knowledge base using **Mistral Embeddings** and **Pinecone Vector Database**. This project uses a microservice architecture where the embedding logic is isolated in a Docker container.

---

## Quick Start

### 1. Prerequisites
* **Docker Desktop** installed and running.
* **Python 3.10+** installed locally.
* API Keys for **Mistral AI** and **Pinecone**.

### 2. Environment Setup
Create a `.env` file in the root directory (use `.env.example` as a template):
```env
PINECONE_API_KEY=your_pinecone_key_here
MISTRAL_API_KEY=your_mistral_key_here
