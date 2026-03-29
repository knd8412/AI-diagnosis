import os
import chromadb
import uuid
from chromadb.utils.embedding_functions import EmbeddingFunction
# Import the existing utility that talks to your internal embedding-service
from llm_integration.retrieval.query_embedder import embed_query

class InternalServiceEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function that routes requests through the 
    internal 'embedding-service' container.
    """
    def __init__(self):
        """Initialize the embedding function."""
        pass  # Fix deprecation warning
    
    def __call__(self, input):
        # input can be a single string or a list of strings
        if isinstance(input, str):
            return [embed_query(input)]
        
        # If multiple texts are sent, embed each one
        # Note: If your service supports batching, you could optimize this call
        return [embed_query(text) for text in input]
    
class PatientMemory:
    def __init__(self):
        # Use the internal service wrapper instead of the direct Mistral class
        self.internal_ef = InternalServiceEmbeddingFunction()
        
        # ADDED: Environment-aware ChromaDB connection
        # In Docker: uses "chromadb" (service name)
        # Locally: uses "localhost" (set via environment variable)
        chromadb_host = os.environ.get("CHROMADB_HOST", "chromadb")
        chromadb_port = int(os.environ.get("CHROMADB_PORT", "8000"))
        
        # Connects to the dedicated ChromaDB container over the network
        self.client = chromadb.HttpClient(host=chromadb_host, port=chromadb_port)

    def _get_collection(self, patient_id):
        """Helper to get/create collection with the correct embedding function."""
        return self.client.get_or_create_collection(
            name=f"patient_{patient_id}",
            embedding_function=self.internal_ef
        )

    def add_interaction(self, patient_id, role, text):
        """Stores text interaction in a patient-specific collection."""
        collection = self._get_collection(patient_id)
        collection.add(
            documents=[text],
            metadatas=[{"role": role}],
            ids=[f"{patient_id}_{uuid.uuid4()}"]
        )

    def search_history(self, patient_id, query, n_results=3):
        """Performs semantic search using the internal embedding service."""
        collection = self._get_collection(patient_id)
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return "\n".join(results['documents'][0])
    
def get_patient_memory():
    return PatientMemory()