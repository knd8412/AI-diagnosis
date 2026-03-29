import os, json, hashlib, requests, time,sys
from pinecone import Pinecone, ServerlessSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration constants
from config import PINECONE_INDEX_NAME, PINECONE_NAMESPACE, PINECONE_REGION, PINECONE_CLOUD, EMBEDDING_DIMENSION, BATCH_SIZE


MISTRAL_DIMENSION = 1024  
DOCKER_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:5000")
TEXT_FIELD = "page_content"

def ingest_jsonl_to_pinecone(data_path: str) -> int:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY")

    pc = Pinecone(api_key=api_key)

    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )

        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            time.sleep(2)

    index = pc.Index(PINECONE_INDEX_NAME)

    def get_mistral_embeddings(texts: list):
        try:
            response = requests.post(
                f"{DOCKER_URL}/embed/batch", 
                json={"texts": texts}, 
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("embeddings", [])
        except Exception as e:
            print(f"Embedding service failed: {e}")
            return None

    def generate_unique_id(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    total_processed = 0

    with open(data_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
        
        for i in range(0, len(lines), BATCH_SIZE):
            batch_lines = lines[i : i + BATCH_SIZE]
            batch_texts, batch_records = [], []

            for line in batch_lines:
                record = json.loads(line)
                text = record.get(TEXT_FIELD)
                
                if not text:
                    continue
                
                record["id"] = record.get("id") or generate_unique_id(text)
                batch_texts.append(text)
                batch_records.append(record)

            embeddings = get_mistral_embeddings(batch_texts)
            
            if embeddings:
                vectors_to_upload = []
                for rec, emb in zip(batch_records, embeddings):
                    meta = rec.get("metadata", {})
                    if isinstance(meta, dict):
                        rec.update(meta)
                        if "metadata" in rec:
                            del rec["metadata"]
                    
                    vectors_to_upload.append({
                        "id": rec["id"],
                        "values": emb,
                        "metadata": {k: v for k, v in rec.items() if k != "id"}
                    })

                index.upsert(vectors=vectors_to_upload, namespace=PINECONE_NAMESPACE)
                total_processed += len(vectors_to_upload)
                print(f"Uploaded {total_processed} records.")

    return total_processed
