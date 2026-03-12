from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import inngest.fast_api
import inngest

from inngest_workflow import inngest_client, sync_knowledge_base

app = FastAPI(
    title="AI Diagnosis Assistant API",
    description="Serves the main API and the Inngest background functions.",
)

# Serve the Inngest functions
inngest.fast_api.serve(
    app,
    inngest_client,
    [sync_knowledge_base],  
)

class SyncRequest(BaseModel):
    user_id: str = "admin"

@app.get("/", tags=["Health"])
def read_root():
    return {"status": "ok", "message": "API is running. Inngest endpoint is at /api/inngest"}

# NEW: Endpoint to trigger the vector database sync manually
@app.post("/api/trigger-sync", tags=["Admin"])
async def trigger_knowledge_sync(request: SyncRequest):
    """
    Triggers the Inngest background job to embed and sync data to Pinecone.
    """
    await inngest_client.send(
        inngest.Event(
            name="knowledge/sync.requested",
            data={
                "triggered_by": request.user_id
            }
        )
    )
    return {
        "status": "success", 
        "message": "Knowledge base sync triggered in the background. Check Inngest dashboard for progress."
    }