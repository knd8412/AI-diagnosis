# Embedder Service

Microservice for generating and managing vector embeddings for medical text using **Mistral AI** embeddings and **Pinecone** vector database.

## Overview

The Embedder is a Flask-based microservice that:
- **Generates embeddings** from medical text using Mistral AI (1024-dimensional vectors)
- **Stores embeddings** in Pinecone for fast semantic search
- **Provides REST API** for embedding operations
- **Runs in Docker** for isolated, reproducible deployments

## Architecture

```
Medical Text
    ↓
Mistral Embeddings (1024-dim vectors)
    ↓
Embedding Service (Flask)
    ↓
REST API (/embed, /embed/batch)
    ↓
Pinecone Vector Database
```

## Quick Start

### 1. Prerequisites
- Docker Desktop installed and running
- Python 3.10+ (for local development)
- Mistral AI API key
- Pinecone API key

### 2. Environment Setup

Create a `.env` file in the root directory:
```env
# Embedding Service
MISTRAL_API_KEY=your_mistral_key_here
PINECONE_API_KEY=your_pinecone_key_here
EMBEDDING_SERVICE_URL=http://localhost:5001

# Optional: for local development
MISTRAL_MODEL=mistral-embed
PINECONE_INDEX=medical-knowledge
```

### 3. Start the Service

Using Docker Compose:
```bash
docker-compose up -d embedding-service
```

The service will be available at: `http://localhost:5001`

### 4. Verify Service is Running

```bash
curl http://localhost:5001/health
```

Expected response:
```json
{
  "status": "healthy",
  "model": "mistral-embed",
  "dimension": 1024
}
```

## Project Structure

```
embedder/
├── embedder.md                      # This file
├── embedding_service.py             # Main Flask service
├── Dockerfile                       # Docker configuration
├── requirements.txt                 # Python dependencies
└── __pycache__/                     # Python cache
```

## API Endpoints

### Health Check
```bash
GET /health
```

Returns service status and model information.

### Single Embedding
```bash
POST /embed
Content-Type: application/json

{
  "text": "Patient has persistent cough and fever"
}
```

Response:
```json
{
  "text": "Patient has persistent cough and fever",
  "embedding": [0.123, -0.456, ...],
  "dimension": 1024
}
```

### Batch Embeddings
```bash
POST /embed/batch
Content-Type: application/json

{
  "texts": [
    "Patient has fever",
    "Patient has cough",
    "Patient has fatigue"
  ]
}
```

Response:
```json
{
  "count": 3,
  "embeddings": [
    {"text": "...", "embedding": [...]},
    {"text": "...", "embedding": [...]},
    {"text": "...", "embedding": [...]}
  ]
}
```

## Features

### ✅ Mistral Embeddings
- **Model**: mistral-embed
- **Dimension**: 1024 (semantic richness)
- **Language**: English optimized
- **Use Case**: Medical text embeddings

### ✅ Pinecone Integration
- **Fast semantic search** over medical knowledge base
- **Scalable** vector storage
- **Index-based retrieval** for O(log n) lookup
- **Metadata filtering** support

### ✅ REST API
- **Stateless** service for easy scaling
- **Error handling** with descriptive messages
- **Health checks** for monitoring
- **Batch processing** for efficiency

## Testing

Run the test suite to verify the service:

```bash
python tests/test_embedder/test_embedding.py
```

Test coverage includes:
- ✅ Health check endpoint
- ✅ Single text embedding
- ✅ Batch embedding processing
- ✅ Error handling (empty text, missing fields)
- ✅ Performance timing
- ✅ ChromaDB integration (optional)

Expected results:
```
Ran 7 tests in 0.498s
OK (skipped=1)
```

## Configuration

### Mistral API
Set in `.env`:
```env
MISTRAL_API_KEY=your_key
```

Get a key from: https://console.mistral.ai/

### Pinecone
Set in `.env`:
```env
PINECONE_API_KEY=your_key
PINECONE_INDEX=medical-knowledge
```

Get a key from: https://app.pinecone.io/

### Docker Environment
The Dockerfile sets:
- **Base**: Python 3.11-slim
- **Port**: 5001 (embedding service)
- **Workers**: Gunicorn with 4 workers
- **Timeout**: 120 seconds per request

## Docker Deployment

### Build Image
```bash
docker build -t embedder:latest embedder/
```

### Run Container
```bash
docker run -d \
  -p 5001:5001 \
  -e MISTRAL_API_KEY=your_key \
  -e PINECONE_API_KEY=your_key \
  --name embedding-service \
  embedder:latest
```

### View Logs
```bash
docker logs -f embedding-service
```

## Performance

Typical performance on medical text:

| Operation | Time | Notes |
|-----------|------|-------|
| Single embedding | 100-200ms | Mistral API call |
| Batch (100 texts) | 3-5s | Efficient batching |
| Health check | <50ms | Local verification |

## Integration Points

The Embedder is used by:
1. **Query Embedder** (`llm_integration/retrieval/query_embedder.py`)
   - Converts patient queries to embeddings
   - HTTP POST to `/embed` endpoint

2. **ChromaDB Client** (`llm_integration/retrieval/chromaClient.py`)
   - Stores patient conversation history
   - Uses embeddings for similarity search

3. **RAG Chains** (`llm_integration/chains/rag_chains.py`)
   - Retrieves relevant medical context
   - Powers diagnosis generation

## Troubleshooting

### Service Won't Start
```bash
# Check Docker is running
docker ps

# View logs
docker logs embedding-service

# Rebuild image
docker-compose down
docker-compose up -d embedding-service
```

### API Rate Limits
If you hit Mistral API limits:
- Wait for rate limit window (usually 1 minute)
- Increase batch size for efficiency
- Consider upgrading Mistral plan

### Connection Issues
```bash
# Test connectivity
curl -v http://localhost:5001/health

# Check network
docker network ls
docker network inspect bridge
```

### Memory Issues
Adjust in docker-compose.yml:
```yaml
services:
  embedding-service:
    deploy:
      resources:
        limits:
          memory: 4G
```

## Next Steps

1. **Monitor embeddings**: Track quality metrics
2. **Fine-tune retrieval**: Optimize Pinecone queries
3. **Cache embeddings**: Reduce API calls
4. **Add authentication**: Secure the API
5. **Scale horizontally**: Deploy multiple instances

## References

- [Mistral AI Documentation](https://docs.mistral.ai/)
- [Pinecone Vector Database](https://docs.pinecone.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Documentation](https://docs.docker.com/)
