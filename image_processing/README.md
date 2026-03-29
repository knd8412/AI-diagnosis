# Image Processing Service

A FastAPI microservice for chest X-ray analysis. It validates uploaded images, runs inference using a pretrained DenseNet121 model, generates Class Activation Maps (CAMs) for visual explainability, and persists results to the db-service.

> This is a student project. Outputs are not clinically validated and must not be used for real medical diagnosis.

## Overview

When an image is uploaded to `/analyze`, the service runs the following pipeline:

1. **LLM validation** — Pixtral-12B (Mistral) checks whether the image is an X-ray
2. **Heuristic validation** — checks image dimensions, contrast, and intensity
3. **Preprocessing** — normalises pixel values, centre-crops, and resizes to 224x224
4. **Inference** — DenseNet121 produces a probability score for each pathology
5. **CAM generation** — produces spatial heatmaps for the top 5 predicted pathologies
6. **Storage** — raw scan saved to `/scans`, results POSTed to db-service

## Model

| Property | Value |
|---|---|
| Architecture | DenseNet121 (TorchXRayVision) |
| Weights | `densenet121-res224-all` |
| Input size | 224 x 224, single channel |
| Output | Probability score (0–1) per pathology |
| Device | CUDA if available, otherwise CPU |

Weights are downloaded automatically at build time from the TorchXRayVision GitHub releases. No manual download is required.

## API Endpoints

### `POST /analyze`

Accepts a multipart file upload. Returns pathology scores, confident findings, CAM heatmaps, and a result ID from the db-service.

**Request:**
```
Content-Type: multipart/form-data
file: <image file — PNG, JPG, or JPEG>
```

**Response (200):**
```json
{
  "pathologies": {
    "Atelectasis": 0.312,
    "Cardiomegaly": 0.874
  },
  "confident_results": {
    "Cardiomegaly": 0.874
  },
  "top_cams": {
    "Cardiomegaly": [[...], ...]
  },
  "primary_embedding": [...],
  "scan_file": "a1b2c3d4-....png",
  "source_filename": "chest.png",
  "result_id": "<MongoDB ObjectId or null>"
}
```

`confident_results` contains only pathologies with a score above **0.5**.  
`top_cams` contains a 224x224 float array per pathology (top 5 by score).

**Error responses:**

| Code | Reason |
|---|---|
| 400 | Empty file, not detected as an X-ray, or invalid image format |
| 500 | Model inference or file save failed |
| 503 | Mistral LLM validation unavailable |

---

### `GET /scans/{filename}`

Returns a previously saved raw scan PNG by filename.

---

### `GET /health`

Returns `{"status": "ok"}`. Used by Docker healthcheck.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MISTRAL_API_KEY` | — | **Required.** API key for Pixtral-12B image validation |
| `DB_SERVICE_URL` | `http://db-service:8002` | URL of the db-service for persisting results |

## Running Locally (via Docker Compose)

The service is configured in `docker-compose.yml` as `image-processing-service` on port **8001**. It depends on `db-service` being healthy before starting.

```bash
docker compose build image-processing-service
docker compose up image-processing-service
```

Test with:
```bash
curl -X POST http://localhost:8001/analyze \
  -F "file=@/path/to/chest_xray.png"
```

## File Structure

```
image-processing/
├── service.py           # FastAPI app and /analyze endpoint
├── models.py            # XRayModel wrapper (predict, features, CAMs)
├── preprocess.py        # Image loading, normalisation, and resizing
├── processing_config.py # Model weights constant and device config
├── utils.py             # validate_xray() and format_output() helpers
├── Dockerfile
└── requirements.txt
```

## Swapping the Model

To use different weights, change `MODEL_WEIGHTS` in `processing_config.py`:

```python
MODEL_WEIGHTS = "densenet121-res224-all"  # change this
```

Valid TorchXRayVision weight strings are listed in the [TorchXRayVision docs](https://mlmed.org/torchxrayvision/). You will also need to update the `wget` line in the Dockerfile to download the new weights file.
