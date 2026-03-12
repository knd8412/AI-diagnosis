import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from fastapi import FastAPI, UploadFile, File
from preprocess import preprocess, load_image
from models import XRayModel
from utils import validate_xray
import numpy as np

# imports for mongoDB saving

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
import json

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
with open("config.json") as f:
    config = json.load(f)

MONGO_URL = config["mongo_url"]
DB_NAME = config["db_name"]
COLLECTION_NAME = config["collection_name"]

app = FastAPI(title="TorchXRayVision Service")

model = XRayModel()

@app.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = load_image(img_bytes)
    validate_xray(img)
    tensor = preprocess(img)
    preds = model.predict(tensor)
    spatial_feats = model.features(tensor) 
    top_cams = model.top_cams(spatial_feats, preds, top_k=5)
    result = {
        "pathologies": {k: float(v) for k, v in preds.items()},
        "confident_results": {k: float(v) for k,v in preds.items() if v > 0.5},
        "top_cams": top_cams,
        "primary_embedding": np.mean(spatial_feats, axis=(1, 2)).flatten().tolist(),
        "spatial_features": spatial_feats.tolist(),
        "created_at": datetime.now(timezone.utc)
    }

    # store result in MongoDB
    await results_collection.insert_one(result)

    return result

@app.get("/health")
def health():
    return {"status": "ok"}
