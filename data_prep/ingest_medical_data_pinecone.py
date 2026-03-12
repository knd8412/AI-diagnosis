import os
from setup_pinecone import ingest_jsonl_to_pinecone

DATA_PATH = os.getenv("DATA_PATH", "data/respiratory_data.jsonl")

if __name__ == "__main__":
    total = ingest_jsonl_to_pinecone(DATA_PATH)
    print(f" Upserted {total} records.")
