import os,sys
import json
from pinecone import Pinecone, ServerlessSpec

import json
from typing import List, Dict, Optional
import time
import requests

from mistralai import Mistral
#from openai import OpenAI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#import config variables directly from config.py for centralized management
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE, EMBEDDING_DIMENSION, DOCKER_EMBEDDING_URL, MISTRAL_API_KEY

class MedicalDiagnosisAssistant:
    """
    A medical diagnosis assistant that uses vector embeddings and Pinecone
    for semantic search of medical conditions based on symptoms.
    """
    
    def __init__(self, use_docker_embeddings: bool = True):
        """
        Initializes the assistant using centralized settings from config.py.
        """
        print("Initializing Medical Diagnosis Assistant...")
        
        # 1. Initialize Pinecone using config values
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self.namespace = PINECONE_NAMESPACE
        
        # 2. Setup embedding settings from config
        self.use_docker = use_docker_embeddings
        self.docker_url = DOCKER_EMBEDDING_URL
        self.embedding_dimension = EMBEDDING_DIMENSION
        
        # 3. Initialize local Mistral client for metadata extraction
        # We use the key directly from environment variables
        self.mistral_client = self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        
        # 4. Logical check for the embedding source
        if self.use_docker:
            print(f"Using Docker embedding service at {self.docker_url}...")
        else:
            print("Using local Mistral client for embeddings...")
        
        # 5. Connect to the actual index
        self._setup_index()
        self.index = self.pc.Index(self.index_name)
        
        print("✓ Assistant initialized successfully!\n")
    
    def _setup_index(self):
        """Create or connect to Pinecone index."""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            time.sleep(5)
            print("✓ Index created successfully")
        else:
            print(f"✓ Connected to existing index: {self.index_name}")
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding vector for a text string."""
        if self.use_docker:
            try:
                response = requests.post(
                    f"{self.docker_url}/embed",
                    json={"text": text},
                    timeout=30
                )
                response.raise_for_status()
                return response.json()["embedding"]
            except requests.exceptions.RequestException as e:
                print(f"Error calling Docker service: {e}")
                print("Falling back to local Mistral client...")
                return self._create_local_embedding(text)
        else:
            return self._create_local_embedding(text)
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Sends a batch of texts to the Docker embedding service."""
        if self.use_docker:
            try:
                response = requests.post(
                    f"{self.docker_url}/embed/batch",
                    json={"texts": texts},
                    timeout=30
                )
                response.raise_for_status()
                return response.json().get("embeddings", [])
            except Exception as e:
                print(f"Batch Docker embedding failed: {e}. Falling back to local batch...")
        
        # Fallback to local processing if Docker fails
        return [self._create_local_embedding(text) for text in texts]
    
    def _create_local_embedding(self, text: str) -> List[float]:
        """Create embedding using Mistral SDK directly with error handling."""
        try:
            response = self.mistral_client.embeddings.create(
                model="mistral-embed",
                inputs=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"CRITICAL ERROR: Local embedding failed: {e}")
            # Return a list of zeros as a last resort so the code doesn't crash
            return [0.0] * self.embedding_dimension
    
    def load_medical_knowledge(self, data_file: str, batch_size: int = 50):
        """
        Load medical knowledge with optimized text creation and robust error logging.
        """
        print(f"\nLoading medical knowledge from {data_file}...")
        
        try:
            with open(data_file, 'r') as f:
                medical_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return
        
        total_records = len(medical_data)
        failed_ids = []  # To track any records that fail at any stage
        
        print(f"Found {total_records} records. Processing in batches of {batch_size}...")

        for i in range(0, total_records, batch_size):
            batch_records = medical_data[i : i + batch_size]
            
            # 1. PRE-CALCULATE SEARCH TEXT (Optimization: Do this once per record)
            # We store it in a temporary list of dicts so we can reuse it in the metadata loop
            batch_data_with_text = []
            for record in batch_records:
                full_text = self._create_searchable_text(record)
                batch_data_with_text.append({
                    "record": record,
                    "full_text": full_text
                })
            
            # 2. CREATE EMBEDDINGS
            try:
                # Extract only the strings for the API call
                texts_to_embed = [item["full_text"] for item in batch_data_with_text]
                batch_embeddings = self.create_embeddings_batch(texts_to_embed)
            except Exception as e:
                print(f"Failed to get embeddings for batch starting at {i}: {e}")
                failed_ids.extend([r.get('id', 'unknown_id') for r in batch_records])
                continue

            # 3. FORMAT VECTORS FOR PINECONE
            vectors_to_upsert = []
            for item, embedding in zip(batch_data_with_text, batch_embeddings):
                record = item["record"]
                vectors_to_upsert.append({
                    'id': record['id'],
                    'values': embedding,
                    'metadata': {
                        'condition': record.get('condition', ''),
                        'symptoms': record.get('symptoms', ''),
                        'treatment': record.get('treatment', ''),
                        'full_text': item["full_text"] # Reusing the pre-calculated string
                    }
                })
            
            # 4. UPLOAD TO PINECONE
            try:
                self.index.upsert(vectors=vectors_to_upsert)
                end_val = min(i + batch_size, total_records)
                print(f"  ✓ Processed and uploaded {end_val}/{total_records} records...")
            except Exception as e:
                print(f"Failed to upsert batch starting at index {i} to Pinecone: {e}")
                failed_ids.extend([r.get('id', 'unknown_id') for r in batch_records])

        # 5. FINAL ERROR LOGGING
        if failed_ids:
            error_log_file = "failed_uploads.txt"
            with open(error_log_file, "w") as f:
                f.write("\n".join(failed_ids))
            print(f"\n! Process complete with errors: {len(failed_ids)} records failed.")
            print(f"Failed record IDs saved to '{error_log_file}'.")
        else:
            print("\n✓ Medical knowledge base loaded successfully without errors!\n")
    
    def _create_searchable_text(self, record: Dict) -> str:
        """Create a searchable text from a medical record."""
        return f"""
Condition: {record.get('condition', '')}
Symptoms: {record.get('symptoms', '')}
Treatment: {record.get('treatment', '')}
        """.strip()
    
    def diagnose(self, patient_symptoms: str, top_k: int = 5) -> List[Dict]:
        """
        Get potential diagnoses based on patient symptoms using manual embedding.
        """

        # 1. Transform raw text to structured metadata
        metadata = self.extract_clinical_metadata(patient_symptoms)
        extracted_symptoms = metadata.get('symptoms', [])
        
        # Use extracted symptoms if possible, otherwise use the raw input
        symptoms_str = ", ".join(extracted_symptoms) if extracted_symptoms else patient_symptoms

        print(f"\n[AI Extracted Symptoms]: {symptoms_str}")

        # 2. Create embedding using the Docker/Mistral service
        query_embedding = self.create_embedding(symptoms_str)
        
        # 3. Prepare query parameters
        query_params = {
            'vector': query_embedding,
            'top_k': top_k,
            'include_metadata': True,
            'namespace': self.namespace 
        }
        

        # 4. Execute search
        results = self.index.query(**query_params)
        
        
        # 5. Format results with De-duplication
        diagnoses = []
        seen_conditions = set() # Track conditions we've already added

        for match in results.get('matches', []):
            meta = match.get('metadata', {})
            condition_name = meta.get('disease', meta.get('condition', 'Unknown Condition'))
            
            # Only add if we haven't seen this specific disease name yet
            if condition_name not in seen_conditions:
                diagnoses.append({
                    'condition': condition_name,
                    'confidence': match.get('score', 0),
                    'symptoms': meta.get('symptoms', 'N/A'),
                    'treatment': meta.get('treatment', meta.get('recommendation', 'N/A'))
                })
                seen_conditions.add(condition_name)
            
            # Optional: Break early if we have enough unique results
            if len(diagnoses) >= 5:
                break
        
        return diagnoses
    
    def add_new_condition(self, condition_data: Dict):
        """
        Add a new medical condition to the knowledge base.
        
        Args:
            condition_data: Dictionary with condition information
        """
        searchable_text = self._create_searchable_text(condition_data)
        embedding = self.create_embedding(searchable_text)
        
        self.index.upsert(vectors=[{
            'id': condition_data['id'],
            'values': embedding,
            'metadata': {
                'condition': condition_data.get('condition', ''),
                'symptoms': condition_data.get('symptoms', ''),
                'treatment': condition_data.get('treatment', ''),
                'full_text': searchable_text
            }
        }])
        
        print(f"✓ Added condition: {condition_data['condition']}")
    
    def update_condition(self, condition_id: str, updated_data: Dict):
        """
        Update an existing condition without losing existing data (Partial Update).
        """
        # 1. Fetch the existing record from Pinecone
        fetch_response = self.index.fetch(ids=[condition_id])
        
        if condition_id not in fetch_response['vectors']:
            print(f"Error: Condition {condition_id} not found.")
            return

        # 2. Get the current metadata
        existing_meta = fetch_response['vectors'][condition_id]['metadata']
        
        # 3. Merge existing data with new updates
        # This ensures fields like 'symptoms' aren't lost if only 'treatment' is updated
        merged_data = {
            'id': condition_id,
            'condition': updated_data.get('condition', existing_meta.get('condition')),
            'symptoms': updated_data.get('symptoms', existing_meta.get('symptoms')),
            'treatment': updated_data.get('treatment', existing_meta.get('treatment'))
        }

        # 4. Re-generate the searchable text and embedding
        # This prevents the "Stale Vector" problem
        searchable_text = self._create_searchable_text(merged_data)
        embedding = self.create_embedding(searchable_text)
        
        # 5. Upsert the merged record
        self.index.upsert(vectors=[{
            'id': condition_id,
            'values': embedding,
            'metadata': {
                **merged_data,
                'full_text': searchable_text
            }
        }])
        print(f"✓ Condition {condition_id} successfully merged and updated.")
    
    def delete_condition(self, condition_id: str):
        """
        Delete a medical condition from the knowledge base.
        
        Args:
            condition_id: ID of the condition to delete
        """
        self.index.delete(ids=[condition_id])
        print(f"✓ Deleted condition: {condition_id}")
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with index statistics
        """
        return self.index.describe_index_stats()
    
    def search_by_condition(self, condition_name: str, top_k: int = 10) -> List[Dict]:
        """
        Search for similar conditions by name.
        
        Args:
            condition_name: Name of the condition to search for
            top_k: Number of similar conditions to return
            
        Returns:
            List of similar conditions
        """
        query_text = f"Condition: {condition_name}"
        return self.diagnose(query_text, top_k=top_k)
    
    def extract_clinical_metadata(self, user_query: str) -> dict:
        """
        TRANSFORM: Converts raw text into structured JSON data.
        Requirement: 'Transform the user query into structured data'
        """
        prompt = f"""
        Extract medical symptoms and relevant patient history from the following text.
        Return the result ONLY as a JSON object with keys: 
        'symptoms' (list of strings),  and 'duration' (string).
        
        Text: "{user_query}"
        """
        
        try:
            response = self.mistral_client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content

            # Robust parsing
            if not content:
                return {"symptoms": [user_query], "duration": "unknown"}
            
            # CLEANING STEP: Remove markdown backticks and extra whitespace
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"Metadata extraction failed: {e}")
            # Fallback so the code doesn't crash
            return {"symptoms": [user_query],  "duration": "unknown"}


def print_diagnosis_results(results: List[Dict]):
    """Pretty print diagnosis results."""
    print("\nPotential Diagnoses:")
    print("=" * 70)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['condition']} ({res['confidence']*100:.1f}% match)")
        
        print("-" * 70)

# ======================
# TEST EXECUTION BLOCK
# ======================
if __name__ == "__main__":
    # 1. Connect to your working Docker service
    assistant = MedicalDiagnosisAssistant(
        use_docker_embeddings=True
    )
    
    # You only need to run this ONCE. After it finishes, you can comment it back out.
    #assistant.load_medical_knowledge('medical_data.json')
    
    # 3. Test the diagnosis
    symptoms = "I have a persistent cough, fever of 101F, and difficulty breathing"
    results = assistant.diagnose(symptoms, top_k=10)
    
    # Use the helper function to print
    if results:
        print_diagnosis_results(results)
    else:
        print("Still no results. Check if Docker logs show errors during loading.")