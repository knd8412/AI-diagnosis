import os

import inngest


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


INNGEST_DEV = _is_truthy(os.getenv("INNGEST_DEV"))
INNGEST_APP_ID = os.getenv("INNGEST_APP_ID", "ai-diagnosis-assistant")
INNGEST_EVENT_KEY = os.getenv("INNGEST_EVENT_KEY", "local_dev_key")
INNGEST_SIGNING_KEY = os.getenv("INNGEST_SIGNING_KEY", "test-signing-key")
INNGEST_API_BASE_URL = os.getenv("INNGEST_API_BASE_URL")
INNGEST_EVENT_API_BASE_URL = os.getenv("INNGEST_EVENT_API_BASE_URL")


inngest_client = inngest.Inngest(
    app_id=INNGEST_APP_ID,
    event_key=INNGEST_EVENT_KEY,
    signing_key=INNGEST_SIGNING_KEY,
    api_base_url=INNGEST_API_BASE_URL,
    event_api_base_url=INNGEST_EVENT_API_BASE_URL,
    is_production=not INNGEST_DEV,
)
