import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "respiratory-knowledge-v1")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "respiratory-namespace")


def get_pinecone_index():
    """
    Connect to existing Pinecone index

    Returns:
        tuple: (index object, namespace string)
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")

    # Initialise Pinecone client
    pc = Pinecone(api_key=api_key)

    # connect to existing index
    index = pc.Index(PINECONE_INDEX_NAME)

    return index, PINECONE_NAMESPACE