# AI Diagnosis Assistant
 **Live Demo:** http://130.162.172.14:8501
 
 **Github Repository:** https://github.kcl.ac.uk/k24066415/5CCSAGAPLosCaballos

A Retrieval-Augmented Generation (RAG) system that assists clinicians in diagnosing respiratory diseases. The system accepts free-text symptom descriptions, retrieves relevant conditions from a medical knowledge base, generates ranked differential diagnoses with explanations, recommends follow-up tests, and optionally analyses chest X-rays using a pre-trained deep learning model.
 
Built by Team Los Caballos — King's College London, 2025/26.
Alma Loeblich, Anna Rachkova, Armita Eslami Nazari, Conor Brennan, Gregory Ceremisin, Arnav Gupta, Kamyar Nadarkhanidinehkaboudi, Omar Kolashinac and Sofia Davis

---
 
## Architecture Overview
 
The application is composed of 10 Docker services orchestrated via Docker Compose:
 
| Service | Description | Port |
|---|---|---|
| `ui` | Streamlit frontend — patient intake, chat, X-ray upload | 8501 |
| `embedding-service` | Mistral embedding API wrapper | 5001 |
| `backend` | FastAPI + Inngest background job handler | 8000 |
| `inngest` | Inngest dev server for workflow orchestration | 8288 |
| `redis` | Short-term chat history and LLM response cache | 6379 |
| `chromadb` | Long-term semantic patient memory | 8002 |
| `db` | PostgreSQL for patient metadata | 5432 |
| `db-service` | FastAPI REST API over MongoDB for X-ray results | 8003 |
| `image-processing-service` | TorchXRayVision X-ray analysis service | 8001 |
| `mongo` | MongoDB for persistent X-ray analysis storage | 27017 |
 
---
 
## Prerequisites
 
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Git
- API keys (see Environment Variables section below)
 
---
 
## Quick Start
 
### 1. Clone the repository
 
```bash
git clone https://github.kcl.ac.uk/k24066415/5CCSAGAPLosCaballos
cd 5CCSAGAPLosCaballos
```
 
### 2. Create your `.env` file
 
Create a `.env` file in the project root. **Never commit this file — it is already in `.gitignore`.**
 
```env

 
# LLM, Embeddings & ChromaDB memory — Mistral
MISTRAL_API_KEY=your_mistral_api_key
 
# Vector database — Pinecone
PINECONE_API_KEY=your_pinecone_api_key
 
# Embedding service URL (internal Docker network)
EMBEDDING_SERVICE_URL=http://embedding-service:5001
 
# Image processing service URL
IMAGE_SERVICE_URL=http://image-processing-service:8001
 
# MongoDB (used by mongo_service)
MONGO_URL=mongodb://mongo:27017
DB_NAME=xray_service
COLLECTION_NAME=analysis_results
 
# DB service URL (internal Docker network)
DB_SERVICE_URL=http://mongo-service:8003
 
# ChromaDB (patient memory)
CHROMADB_HOST=chromadb
CHROMADB_PORT=8000
 
# Inngest workflow orchestration
INNGEST_EVENT_KEY=your_inngest_event_key
INNGEST_SIGNING_KEY=your_inngest_signing_key
INNGEST_API_BASE_URL=http://inngest:8288
INNGEST_EVENT_API_BASE_URL=http://inngest:8288
INNGEST_APP_ID=ai-diagnosis-assistant
INNGEST_DEV=false
 
# ClearML experiment tracking (optional for evaluation)
CLEARML_API_HOST=https://app.5ccsagap.er.kcl.ac.uk/
CLEARML_WEB_HOST=https://api.5ccsagap.er.kcl.ac.uk
CLEARML_FILES_HOST=https://files.5ccsagap.er.kcl.ac.uk
CLEARML_API_ACCESS_KEY=your_clearml_access_key
CLEARML_API_SECRET_KEY=your_clearml_secret_key
 
# Data path for Pinecone ingestion (file should exist before running ingestion)
DATA_PATH=data/respiratory_data.jsonl
 
# Test configuration
SKIP_CHROMADB_TESTS=0
```
 
### 3. Build and run
 
```bash
docker compose up -d --build
```
 
The first build will take 15–30 minutes as the TorchXRayVision model weights are downloaded automatically.
 
### 4. Open the app
 
Navigate to `http://localhost:8501` or `http://127.0.0.1:8501/` in your browser.
 
---

## Project Structure
 
```
.
├── __init__.py                     # Package initialization
├── __pycache__/                    # Python bytecode cache
├── config.py                       # Shared configuration (API keys, settings)
├── docker-compose.yml              # All service definitions and orchestration
├── pyrightconfig.json              # Pyright type checking configuration
├── README.md                        # This file
├── main.py                         # FastAPI app entry point (backend)
├── requirements.txt                # Python dependencies
├── requirements-dev.txt            # Development dependencies (pytest, etc.)
├── data/
│   └── respiratory_data.jsonl      # Medical knowledge base (JSONL format, not included in repo)
├── db_data/                        # PostgreSQL persistent storage
├── chroma_data/                    # ChromaDB persistent storage
├── patientIDdb/                    # Patient ID database
├── data_prep/
│   ├── ingest_medical_data_pinecone.py
│   ├── setup_pinecone.py           # Ingests JSONL data into Pinecone vector DB
│   └── __pycache__/
├── SQLdb/
│   ├── __init__.py
│   ├── models.py                   # SQLAlchemy patient data models
│   └── __pycache__/
├── mongo_service/
│   ├── __init__.py
│   ├── dockerfile                  # Dockerfile for MongoDB REST service
│   ├── requirements.txt            # FastAPI, motor (async MongoDB driver)
│   ├── service.py                  # FastAPI REST API for MongoDB X-ray results
│   └── __pycache__/
├── embedder/
│   ├── __init__.py
│   ├── Dockerfile                  # Flask service container
│   ├── embedding_service.py        # Mistral embedding API wrapper
│   ├── embedder.md                 # Service documentation
│   ├── requirements.txt            # Flask, requests, python-dotenv
│   └── __pycache__/
├── evaluation/
│   ├── __init__.py
│   ├── Evaluation.md               # Evaluation methodology documentation
│   ├── collect_data.py             # Collects RAG pipeline outputs for evaluation
│   ├── eval.py                     # RAGAS (RAG Assessment) evaluation script
│   ├── eval_data.json              # Ground truth evaluation dataset (15 queries)
│   ├── ragas_results.csv           # RAGAS evaluation metrics output
│   └── __pycache__/
├── image_processing/
│   ├── __init__.py
│   ├── Dockerfile                  # PyTorch service container
│   ├── manual_tests/
│   │   ├── test_api.py             # Manual API testing script
│   │   └── test_torchxray.ipynb    # Jupyter notebook for X-ray model testing
│   ├── models.py                   # TorchXRayVision model wrapper
│   ├── preprocess.py               # Image preprocessing (normalization, resizing)
│   ├── processing_config.py        # Configuration for image processing
│   ├── requirements.txt            # FastAPI, torchxrayvision, opencv, torch
│   ├── service.py                  # FastAPI X-ray analysis endpoint
│   ├── utils.py                    # Utility functions for image handling
│   └── __pycache__/
├── inngest_workflow/
│   ├── __init__.py
│   ├── Dockerfile                  # Node.js service container for Inngest
│   ├── README.md                   # Inngest workflow documentation
│   ├── client.py                   # Inngest client initialization
│   ├── functions.py                # Inngest background job functions (Pinecone sync)
│   ├── requirements.txt            # inngest, python-dotenv
│   └── __pycache__/
├── llm_integration/
│   ├── __init__.py
│   ├── llm_integration.md          # LLM integration architecture documentation
│   ├── requirements.txt            # langchain, openrouter-python, pinecone
│   ├── chains/
│   │   ├── __init__.py
│   │   ├── diagnosis_orchestration.py  # Orchestrates RAG pipeline steps
│   │   ├── explanation_chain.py    # Generates explanations for diagnoses
│   │   ├── image_analysis_chain.py # Analyzes X-ray findings
│   │   ├── rag_chains.py           # Main RAG pipeline (retrieve + generate)
│   │   ├── symptom_extraction_chain.py  # Extracts symptoms from free text
│   │   └── __pycache__/
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── diagnosis_prompts.py    # LangChain prompt templates
│   │   └── __pycache__/
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── chromaClient.py         # ChromaDB client for patient memory
│   │   ├── pinecone_client.py      # Pinecone vector DB connection
│   │   ├── query_embedder.py       # Calls embedding service for queries
│   │   └── __pycache__/
│   ├── service/
│   │   ├── __init__.py
│   │   ├── client.py               # LLM client (OpenRouter via LangChain)
│   │   ├── image_processing_client.py  # Client for image processing service
│   │   └── __pycache__/
│   └── tests/
│       ├── __init__.py
│       ├── test_diagnosis_orchestr.py  # Tests diagnosis orchestration
│       ├── test_prompts.py         # Tests prompt formatting
│       ├── test_rag.py             # Tests RAG pipeline
│       └── __pycache__/
├── tests/                          # Integration and unit tests
│   ├── __init__.py
│   ├── __pycache__/
│   ├── system_evaluation/
│   │   └── test_evaluation_service.py  # Tests RAG evaluation service
│   ├── test_chromadb/
│   │   ├── __init__.py
│   │   ├── test_chromadb.py        # Tests ChromaDB patient memory
│   │   └── __pycache__/
│   ├── test_clearML/
│   │   ├── __init__.py
│   │   ├── test_clearml.py         # Tests ClearML integration
│   │   └── __pycache__/
│   ├── test_embedder/
│   │   ├── __init__.py
│   │   ├── test_embedding.py       # Tests embedding service API
│   │   └── __pycache__/
│   ├── test_evaluation/
│   │   ├── __init__.py
│   │   ├── test_evaluation.py      # Tests RAG evaluation metrics
│   │   └── __pycache__/
│   ├── test_image_processing/
│   │   ├── __init__.py
│   │   ├── test_image_preprocessing.py  # Tests image preprocessing
│   │   ├── test_models.py          # Tests TorchXRayVision model
│   │   ├── test_utils.py           # Tests utility functions
│   │   └── __pycache__/
│   ├── test_inngest/
│   │   ├── __init__.py
│   │   ├── test_inngest_endpoint.py  # Tests Inngest workflow endpoints
│   │   ├── test_inngest_function.py  # Tests Inngest background functions
│   │   ├── test_pinecone_safety.py # Tests Pinecone safety constraints
│   │   └── __pycache__/
│   ├── test_langchain/
│   │   ├── __init__.py
│   │   ├── test_langchain.py       # Tests LangChain prompt and chain integration
│   │   └── __pycache__/
│   ├── test_mongodb/
│   │   ├── __init__.py
│   │   ├── test_mongodb.py         # Tests MongoDB service
│   │   └── __pycache__/
│   └── test_ui/
│       ├── __init__.py
│       ├── test_ui.py              # Tests Streamlit UI functionality
│       └── __pycache__/
├── ui/
│   ├── Dockerfile.ui               # Streamlit service container
│   ├── requirements.txt            # streamlit, pandas, requests
│   ├── styles.css                  # Custom CSS styling
│   ├── ui.py                       # Main Streamlit application (patient interface)
│   └── __pycache__/
```
 
---
 
## Data Ingestion (First-Time Setup)
 
The medical knowledge base needs to be ingested into Pinecone before the RAG pipeline works. This is triggered automatically via Inngest when the backend starts. To trigger it manually:
 
```bash
docker exec -it 5ccsagaploscaballos-backend-1 python data_prep/setup_pinecone.py
```

**Note**: You must first place your medical knowledge base file at `data/respiratory_data.jsonl`. This file is not included in the repository. The expected format is JSONL with one JSON object per line, containing respiratory condition data.
 
The example dataset covers respiratory conditions including Bronchial Asthma, Pneumonia, Tuberculosis, Cystic Fibrosis, and Common Cold.
 
---
 

## Testing
 
```bash
# Run all tests
python -m pytest tests/
---

## API Endpoints

### Backend Service (http://localhost:8000)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check |
| `/api/diagnose` | POST | Generate diagnosis from symptoms |
| `/trigger-sync` | POST | Manually trigger Pinecone sync via Inngest |

**POST /api/diagnose**
```json
Request:
{
  "patient_id": "patient_001",
  "symptoms": "I have persistent cough and fever"
}

Response:
{
  "diagnosis": "Likely conditions...",
  "confidence": 0.85,
  "test_recommendations": ["Chest X-ray", "CBC"],
  "retrieved_context": [...]
}
```

### Image Processing Service (http://localhost:8001)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Service health check |
| `/analyze` | POST | Analyze X-ray image using TorchXRayVision |
| `/scans/{filename}` | GET | Retrieve saved scan metadata |

**POST /analyze**
```json
Request: Multipart form-data with image file

Response:
{
  "pathologies": ["pneumonia", "bronchitis"],
  "confident_results": ["pneumonia"],
  "scan_id": "scan_abc123",
  "created_at": "2025-03-23T10:30:00Z"
}
```

### MongoDB Service (http://localhost:8003)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Service health check |
| `/results` | POST | Save X-ray analysis results |
| `/results/{result_id}` | GET | Retrieve saved result |
| `/chatlogs` | POST | Save chat session logs |

**POST /results**
```json
Request:
{
  "patient_id": "patient_001",
  "pathologies": ["pneumonia"],
  "scan_file": "scan_abc123.png",
  "top_cams": {...}
}

Response:
{
  "result_id": "result_xyz789",
  "created_at": "2025-03-23T10:30:00Z"
}
```

### Embedding Service (http://localhost:5001)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Service health check |
| `/embed` | POST | Generate embedding for single text |
| `/embed/batch` | POST | Generate embeddings for multiple texts |

**POST /embed**
```json
Request:
{
  "text": "Patient has persistent cough"
}

Response:
{
  "embedding": [0.123, -0.456, ...],  # 1024-dimensional vector
  "model": "mistral-embed"
}
```

=======
## Testing

```bash
# Core CI tests
python -m pytest -q tests/test_inngest/test_inngest_function.py
python -m unittest -v tests.test_inngest.test_inngest_endpoint tests.test_inngest.test_pinecone_safety

# Optional external script tests (requires API keys/services)
python llm_integration/tests/test_prompts.py
python llm_integration/tests/test_rag.py
python llm_integration/tests/test_diagnosis_orchestr.py

# Optional notebook execution test
jupyter nbconvert --to notebook --execute image_processing/manual_tests/test_torchxray.ipynb --output test_torchxray.executed.ipynb --ExecutePreprocessor.timeout=1800
```
 
---

## Database Schemas

### SQLite (PostgreSQL) - Patient Metadata

**Table: patients**
```sql
CREATE TABLE patients (
  patient_id VARCHAR PRIMARY KEY,
  name VARCHAR NOT NULL,
  age INTEGER,
  gender VARCHAR,
  medical_history TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### MongoDB - X-Ray Analysis Results

**Collection: analysis_results**
```json
{
  "_id": ObjectId,
  "patient_id": "patient_001",
  "pathologies": ["pneumonia", "bronchitis"],
  "confident_results": ["pneumonia"],
  "scan_file": "scan_abc123.png",
  "top_cams": {
    "cxr_image_data": {...},
    "locations": [...]
  },
  "scan_id": "scan_abc123",
  "created_at": ISODate("2025-03-23T10:30:00Z")
}
```

### ChromaDB - Patient Memory

**Collection: patient_memory_{patient_id}**
```
Stores embeddings of patient interactions for semantic memory:
- Documents: Previous symptom descriptions, diagnoses, follow-up notes
- Metadata: timestamps, session_id, interaction_type
- Purpose: Long-term patient context for RAG retrieval
```

### Pinecone - Medical Knowledge Base

**Index: respiratory-knowledge-v1**
```
Namespace: respiratory-namespace
Vector dimension: 1024 (Mistral embeddings)
Metadata per vector:
{
  "condition_name": "pneumonia",
  "page_content": "Description of condition...",
  "source": "medical_database",
  "created_at": "2025-01-01"
}
```

### Redis - Chat History Cache

**Key Format: `patient_{patient_id}:session_{session_id}`**
```
Data structure: Hash
Fields:
  - message_1: "User message"
  - response_1: "Assistant response"
  - timestamp_1: 1234567890
TTL: 3600 seconds (1 hour)
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. Port Already in Use
```bash
# Check which process is using a port (e.g., 8501)
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Kill the process or change port in docker-compose.yml
```

#### 2. Docker Connection Refused
```bash
# Ensure Docker daemon is running
docker ps

# Restart Docker service
sudo systemctl restart docker  # Linux
# Use Docker Desktop UI on Windows/macOS
```

#### 3. Cannot Find data/respiratory_data.jsonl
- This file is NOT included in the repository
- You must provide your own JSONL file with medical knowledge base
- Format: One JSON object per line with condition data
- Path: `data/respiratory_data.jsonl`

#### 4. API Key Validation Failures
- Verify keys are correct in `.env` file
- Check OPENROUTER_API_KEY, MISTRAL_API_KEY, PINECONE_API_KEY
- Ensure keys have proper permissions/access levels
- Common: keys with leading/trailing whitespace

#### 5. ChromaDB Connection Refused (Port 8002)
```bash
# Check if ChromaDB container is running
docker ps | grep chromadb

# Restart ChromaDB
docker compose up -d chromadb

# Set SKIP_CHROMADB_TESTS=1 if unavailable for testing
```

#### 6. Out of Memory During First Build
- TorchXRayVision model weights are ~500MB+
- First build downloads large ML models
- Allocate 4GB+ RAM to Docker
- On Docker Desktop: Settings → Resources → Memory

#### 7. Pinecone Ingestion Fails
```bash
# Manually trigger ingestion
docker exec -it 5ccsagaploscaballos-backend-1 python data_prep/setup_pinecone.py

# Check logs
docker logs 5ccsagaploscaballos-backend-1
```

#### 8. RAG Pipeline Returns Empty Results
- Verify Pinecone is initialized: check Pinecone dashboard
- Verify data file exists: `data/respiratory_data.jsonl`
- Check PINECONE_API_KEY and index name are correct
- Verify knowledge base has relevant medical data

#### 9. Image Processing Service Timeout
- First X-ray analysis loads TorchXRayVision model (slow)
- Subsequent requests are faster (cached in memory)
- Ensure adequate disk space for model weights

#### 10. MongoDB Connection Issues
```bash
# Check MongoDB is running
docker exec -it 5ccsagaploscaballos-mongo-1 mongo

# Verify MONGO_URL and DB_NAME in .env
```

#### 11. Test Failures
```bash
# Run with verbose output
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_embedder/test_embedding.py::TestEmbeddingService::test_health

# Skip ChromaDB tests if unavailable
SKIP_CHROMADB_TESTS=1 python -m pytest tests/
```

---

## Development Setup

### Local Development (Without Docker)

For faster iteration during development, you can run services locally:

#### Prerequisites
- Python 3.9+
- pip or conda
- PostgreSQL (local instance)
- Redis (local instance or via Docker)
- MongoDB (local instance or via Docker)

#### Setup

1. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. **Install development dependencies**
```bash
pip install -r requirements-dev.txt
pip install -r requirements.txt
```

3. **Set up services (recommended via Docker)**
```bash
# Run only backend infrastructure
docker compose up -d postgres redis mongo chromadb
# Skip UI, backend, embedding-service, image-processing-service

# Run backend locally
python main.py

# Run embedding service locally (another terminal)
cd embedder
python embedding_service.py

# Run image processing service locally (another terminal)
cd image_processing
python service.py
```

4. **Initialize databases**
```bash
# Create patient table
python SQLdb/models.py

# Ingest Pinecone data
python data_prep/setup_pinecone.py
```

### IDE Setup

#### VS Code
1. Install Python extension (ms-python.python)
2. Install Pylance (ms-python.vscode-pylance)
3. Create `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black"
}
```

#### PyCharm
1. Configure interpreter: Settings → Project → Python Interpreter
2. Set up virtual environment as interpreter
3. Enable Django framework support (optional)

### Code Style & Linting

```bash
# Format code with Black
black .

# Run linting
pylint llm_integration/ embedder/ image_processing/

# Type checking with Pyright
pyright

# Run tests with coverage
pytest tests/ --cov=. --cov-report=html
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "Description of changes"

# Push to remote
git push origin feature/your-feature-name

# Create pull request to dev branch
```

### Debugging

#### Backend (FastAPI)
```bash
# Run with debug logging
DEBUG=1 python main.py

# Use debugger breakpoints in VS Code
# Set breakpoint, then Run → Start Debugging
```

#### RAG Pipeline
```python
# In test script
from llm_integration.chains.rag_chains import DiagnosisRAG

rag = DiagnosisRAG()
result = rag.diagnose(patient_id="test", query="symptoms...")
print(result)  # Inspect result structure
```

#### Image Processing
```bash
# Test single image
cd image_processing
python -c "from models import XRayModel; model = XRayModel(); model.analyze('path/to/image.png')"
```

---

## Deployment (Oracle Cloud)

The application is deployed on an Oracle Cloud free-tier Ubuntu VM.
http://130.162.172.14:8501

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable deployable version |
| `develop` | Active development |
| `release/release-v*` | Release candidates |
| Feature/Bugfix branches | Merged into dev via pull request |
 
Always run `docker compose down --remove-orphans` before switching branches to avoid port conflicts.
 
---
 
## Team
 
Los Caballos — King's College London, 5CCSAGAP Large Group Project, 2025/26.
