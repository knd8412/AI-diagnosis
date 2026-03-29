import sys
import os
import base64
import numpy as np
import httpx
import uuid

# =============================================================
# IMAGE PROCESSING SERVICE
# =============================================================
# Description:
#   Provides HTTP API endpoints for processing medical X-ray images:
#   - /analyze: receives an uploaded image, validates via LLM, runs model inference,
#     saves scan to disk, and stores results via db-service
#   - /scans/{filename}: returns stored scan files
#   - /health: basic service healthcheck
# =============================================================

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from preprocess import preprocess, load_image
from models import XRayModel
from mistralai import Mistral

mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

DB_SERVICE_URL = os.getenv("DB_SERVICE_URL", "http://db-service:8002")
SCAN_DIR = "/scans"
os.makedirs(SCAN_DIR, exist_ok=True)

app = FastAPI(title="TorchXRayVision Service")
model = XRayModel()


# is_xray_llm is a heuristic check using an external LLM.
async def is_xray_llm(img_bytes: bytes) -> bool:
    b64 = base64.b64encode(img_bytes).decode()
    
    response = mistral_client.chat.complete(
        model="pixtral-12b-2409",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "Is this a medical X-ray image? Answer YES, even if it looks a little bit like an x-ray, false positive is much better than false negatives. Reply with only YES or NO."}
            ]
        }],
        max_tokens=5
    )
    
    answer = response.choices[0].message.content.strip().upper()
    return answer == "YES"


@app.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    # This endpoint covers the full pipeline for one uploaded image:
    #   1. convert upload to bytes
    #   2. LLM-based X-ray detection
    #   3. save raw scan to disk
    #   4. preprocess + model inference
    #   5. optional storage in db-service
    try:
        img_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {e}")

    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file upload")

    try:
        is_xray = await is_xray_llm(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM checking failed: {e}")

    if not is_xray:
        raise HTTPException(status_code=400, detail="Uploaded image is not detected as an X-ray.")

    scan_id = str(uuid.uuid4())
    scan_filename = f"{scan_id}.png"
    scan_path = os.path.join(SCAN_DIR, scan_filename)

    try:
        with open(scan_path, "wb") as f:
            f.write(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save scan file: {e}")

    try:
        img = load_image(img_bytes)
        tensor = preprocess(img)
        preds = model.predict(tensor)
        spatial_feats = model.features(tensor)
        top_cams = model.top_cams(spatial_feats, preds, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")

    result = {
        "pathologies": {k: float(v) for k, v in preds.items()},
        "confident_results": {k: float(v) for k, v in preds.items() if v > 0.5},
        "top_cams": top_cams,
        "primary_embedding": np.mean(spatial_feats, axis=(1, 2)).flatten().tolist(),
        "scan_file": scan_filename,
        "source_filename": file.filename,
    }

    async with httpx.AsyncClient() as http:
        try:
            db_response = await http.post(f"{DB_SERVICE_URL}/results", json=result, timeout=5)
            db_response.raise_for_status()
            result["result_id"] = db_response.json().get("id")
        except Exception as e:
            result["result_id"] = None
            print(f"[image-service] Failed to save result to db-service: {e}")

    return JSONResponse(status_code=200, content=result)


@app.get("/scans/{filename}")
async def get_scan(filename: str):
    path = f"{SCAN_DIR}/{filename}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Scan not found")
    return FileResponse(path, media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok"}