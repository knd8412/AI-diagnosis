# Inngest Pipeline Guide

This folder contains the Inngest client + function that syncs prepared respiratory data into Pinecone.

## Required env vars
- `PINECONE_API_KEY`
- `MISTRAL_API_KEY`
- `OPENROUTER_API_KEY` (UI/LLM usage)

### Inngest (local)
- `INNGEST_EVENT_KEY=local_dev_key`
- `INNGEST_SIGNING_KEY=signkey-test-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef`
- `INNGEST_DEV=1`
- `INNGEST_API_BASE_URL=http://inngest:8288` (when running in Docker Compose)
- `INNGEST_EVENT_API_BASE_URL=http://inngest:8288` (when running in Docker Compose)

### Inngest (production)
Use real keys from Inngest dashboard:
- Event key: https://www.inngest.com/docs/events/creating-an-event-key
- Signing key: https://www.inngest.com/docs/platform/signing-keys

## Start locally
```bash
docker compose up --build
```

Endpoints:
- Backend: `http://localhost:8000`
- Inngest dev: `http://localhost:8288`

## Trigger sync
```bash
curl -X POST http://localhost:8000/api/trigger-sync \
  -H "Content-Type: application/json" \
  -d '{"user_id":"manual-test"}'
```

Check run status in Inngest UI.

## Test suite
From repo root:
```bash
pip3 install -r requirements.txt
python3 -m unittest discover -s tests -v
```

Covers:
- API trigger sends `knowledge/sync.requested`
- Function validates file and runs ingest step
- Pinecone ingest does not delete/recreate an existing index

## Pinecone safety behavior
`data_prep/setup_pinecone.py` now **preserves existing index** and only creates it if missing.
