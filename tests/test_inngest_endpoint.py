import unittest
import os
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

os.environ.setdefault("INNGEST_SIGNING_KEY", "test-signing-key")
os.environ.setdefault("INNGEST_EVENT_KEY", "test-event-key")
os.environ.setdefault("INNGEST_DEV", "1")

import main


class TriggerSyncEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_trigger_sync_sends_inngest_event(self):
        with patch("main.inngest_client.send", new=AsyncMock()) as mock_send:
            response = self.client.post("/api/trigger-sync", json={"user_id": "qa-user"})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["status"], "success")
            mock_send.assert_awaited_once()

            event_arg = mock_send.await_args.args[0]
            self.assertEqual(event_arg.name, "knowledge/sync.requested")
            self.assertEqual(event_arg.data["triggered_by"], "qa-user")


if __name__ == "__main__":
    unittest.main()
