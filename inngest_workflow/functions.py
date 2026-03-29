import os
import inngest 
from .client import inngest_client 

# Import your actual Pinecone ingestion logic
from data_prep.setup_pinecone import ingest_jsonl_to_pinecone

# Path to the processed JSONL data
DATA_PATH = os.getenv("DATA_PATH", "data/respiratory_data.jsonl")

async def _sync_knowledge_base_impl(step, data_path: str = DATA_PATH, ingest_fn=ingest_jsonl_to_pinecone):
    """
    Reads the processed JSONL respiratory data, generates embeddings, 
    and upserts them to Pinecone safely in the background.
    """

    # --- STEP 1: VALIDATE DATA EXISTS ---
    def check_file():
        if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:
            raise FileNotFoundError(f"Data file not found or empty at: {data_path}. Run data prep first.")
        return data_path

    file_path = await step.run("validate-data-file", check_file)


    # --- STEP 2: EMBED AND UPSERT TO PINECONE ---
    # By wrapping this in step.run, Inngest will automatically retry 
    # this specific block if Pinecone or Mistral APIs fail.
    def do_ingest():
        # Calls your existing script logic
        total_upserted = ingest_fn(file_path)
        return total_upserted

    upserted_total = await step.run("embed-and-upsert-to-pinecone", do_ingest)


    # --- STEP 3: RETURN STATUS ---
    return {
        "status": "success",
        "file_used": file_path,
        "records_processed": upserted_total,
        "message": f"Successfully synced {upserted_total} medical records to the vector database."
    }


@inngest_client.create_function(
    fn_id="sync-knowledge-base",
    name="Sync Medical Knowledge Base to Pinecone",
    trigger=inngest.TriggerEvent(event="knowledge/sync.requested"),
)
async def sync_knowledge_base(ctx):
    return await _sync_knowledge_base_impl(ctx.step)
