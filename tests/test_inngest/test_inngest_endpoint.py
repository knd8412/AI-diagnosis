import unittest
import os
import asyncio
from unittest.mock import AsyncMock, patch

os.environ.setdefault("INNGEST_SIGNING_KEY", "test-signing-key")
os.environ.setdefault("INNGEST_EVENT_KEY", "test-event-key")
os.environ.setdefault("INNGEST_DEV", "1")

import main


class TriggerSyncEndpointTests(unittest.TestCase):
    def test_trigger_sync_sends_inngest_event(self):
        with patch("main.inngest_client.send", new=AsyncMock()) as mock_send:
            response = asyncio.run(
                main.trigger_knowledge_sync(main.SyncRequest(user_id="qa-user"))
            )

            payload = response
            self.assertEqual(payload["status"], "success")
            mock_send.assert_awaited_once()

            event_arg = mock_send.await_args.args[0]
            self.assertEqual(event_arg.name, "knowledge/sync.requested")
            self.assertEqual(event_arg.data["triggered_by"], "qa-user")


if __name__ == "__main__":
    unittest.main()
