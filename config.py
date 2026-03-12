import os
from dotenv import load_dotenv

# Load once here so other files don't necessarily have to
load_dotenv()

# Pinecone Infrastructure
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "respiratory-knowledge-v1")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "respiratory-namespace")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1") 
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")

# Embedding Settings
EMBEDDING_DIMENSION = 1024
DOCKER_EMBEDDING_URL = "http://localhost:5000"
TEXT_FIELD = "page_content"

# Batching
BATCH_SIZE = int(os.getenv("PINECONE_BATCH_SIZE", "90"))

# Mistral API
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')