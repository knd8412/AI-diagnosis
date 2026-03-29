import requests
import os

def embed_query(query):
    # Environment-aware URL
    embedding_host = os.environ.get("EMBEDDING_SERVICE_HOST", "embedding-service")
    embedding_port = os.environ.get("EMBEDDING_SERVICE_PORT", "5001")
    embedding_url = f"http://{embedding_host}:{embedding_port}/embed"
    
    try:
        response = requests.post(
            embedding_url,
            json={"text": query},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Cannot connect to embedding service at {embedding_url}. "
            f"Ensure the container is running. Error: {e}"
        )