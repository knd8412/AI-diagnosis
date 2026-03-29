from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from datetime import datetime, timezone
import os

# =============================================================
# MONGO SERVICE
# =============================================================
# Description:
#   Lightweight MongoDB CRUD API for image analysis results and chat logs.
#   Includes endpoint helpers and simple ObjectId serialization.
# =============================================================

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "xray_service")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "analysis_results")

client: AsyncIOMotorClient = None
db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Function note:
    # - sets client and db on startup, closes connection on shutdown
    global client, db
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]
    yield
    client.close()

app = FastAPI(title="DB Service", lifespan=lifespan)

def serialise(doc: dict) -> dict:
    # Function note:
    # - converts MongoDB ObjectId to string for JSON serialization
    # - mutates in-place for simplicity; clone if immutability is needed
    doc["_id"] = str(doc["_id"])
    return doc

# --- Analysis Results ---
@app.post("/results")
async def save_result(result: dict):
    # Function note:
    # - appends creation metadata and saves result document
    # - does not validate fields (assumes calling service has done types checks)
    result["created_at"] = datetime.now(timezone.utc)
    res = await db[COLLECTION_NAME].insert_one(result)
    return {"id": str(res.inserted_id)}

@app.get("/results/{result_id}")
async def get_result(result_id: str):
    doc = await db[COLLECTION_NAME].find_one({"_id": ObjectId(result_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Result not found")
    return serialise(doc)

async def fetch_field(result_id: str, field: str):
    # Function note:
    # - utility shared by endpoint getters for modular code
    # - translates not-found into 404
    # - if requested field is missing by schema, may raise KeyError (intentionally strict)
    try:
        object_id = ObjectId(result_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid result_id: {e}")

    doc = await db[COLLECTION_NAME].find_one(
        {"_id": object_id},
        {field: 1, "_id": 0}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Result not found")
    if field not in doc:
        raise HTTPException(status_code=404, detail=f"Field '{field}' not found")
    return doc[field]

@app.get("/results/{result_id}/pathologies")
async def get_pathologies(result_id: str):
    return await fetch_field(result_id, "pathologies")

@app.get("/results/{result_id}/top_cams")
async def get_top_cams(result_id: str):
    return await fetch_field(result_id, "top_cams")

@app.get("/results/{result_id}/confident_results")
async def get_confident_results(result_id: str):
    return await fetch_field(result_id, "confident_results")

@app.get("/results/{result_id}/scan_file")
async def get_scan_file(result_id: str):
    return await fetch_field(result_id, "scan_file")

@app.get("/results/{result_id}/{field}")
async def get_other(result_id: str, field: str):
    return await fetch_field(result_id, field)

# --- Chat Logs ---
@app.post("/chatlogs")
async def save_chatlog(log: dict):
    # Function note:
    # - upserts chat log by session_id to preserve continuity
    # - keeps created_at when record exists and updates updated_at
    # - includes join with any result_ids in a set semantics
    session_id = log.get("session_id")
    result_ids = log.pop("result_ids", [])  # remove from log to handle separately

    update = {
        "$set": {**log, "updated_at": datetime.now(timezone.utc)},
        "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
    }
    if result_ids:
        update["$addToSet"] = {"result_ids": {"$each": result_ids}}

    await db["chat_logs"].update_one(
        {"session_id": session_id},
        update,
        upsert=True
    )
    return {"status": "ok"}

@app.get("/chatlogs/{session_id}/messages")
async def get_messages(session_id: str):
    doc = await db["chat_logs"].find_one(
        {"session_id": session_id},
        {"messages": 1, "_id": 0}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Chat log not found")
    return doc["messages"]

@app.get("/chatlogs/by-patient/{patient_id}")
async def get_chatlogs_by_patient(patient_id: str):
    cursor = db["chat_logs"].find(
        {"patient_id": patient_id},
        {"session_id": 1, "messages": 1, "result_ids": 1, "created_at": 1, "_id": 0}
    ).sort("created_at", -1)
    docs = await cursor.to_list(length=100)
    return docs

@app.get("/chatlogs/{session_id}")
async def get_chatlog(session_id: str):
    doc = await db["chat_logs"].find_one({"session_id": session_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Chat log not found")
    return serialise(doc)

@app.get("/health")
def health():
    return {"status": "ok"}
