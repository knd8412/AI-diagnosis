import requests

# Ensure this matches the service name in docker-compose.yml
DOCKER_URL = "http://embedding-service:5001"

def embed_query(query_text: str):
    try:
        # FIX: The endpoint must be /embed, not the root URL
        response = requests.post(
            f"{DOCKER_URL}/embed", 
            json={"text": query_text},
            timeout=10
        )
        response.raise_for_status()
        return response.json()["embedding"]

    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to embedding service. Ensure the container is running.")
    except Exception as e:
        raise RuntimeError(f"Embedding service failed: {e}")