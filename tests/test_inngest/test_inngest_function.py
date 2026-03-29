import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from inngest_workflow.functions import _sync_knowledge_base_impl, sync_knowledge_base


class FakeStep:
    def __init__(self):
        self.step_ids = []

    async def run(self, step_id, fn):
        self.step_ids.append(step_id)
        return fn()


class InngestFunctionTests(unittest.TestCase):
    def test_sync_function_uses_ctx_step(self):
        ctx = Mock()
        ctx.step = object()

        expected = {"status": "success", "records_processed": 1}
        with patch(
            "inngest_workflow.functions._sync_knowledge_base_impl",
            new=AsyncMock(return_value=expected),
        ) as impl_mock:
            result = asyncio.run(sync_knowledge_base._handler(ctx))

        impl_mock.assert_awaited_once_with(ctx.step)
        self.assertEqual(result, expected)

    def test_sync_impl_validates_file_and_ingests(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write('{"page_content": "sample"}\n')
            data_path = tmp.name

        ingest_fn = Mock(return_value=7)
        step = FakeStep()

        try:
            result = asyncio.run(_sync_knowledge_base_impl(step, data_path=data_path, ingest_fn=ingest_fn))
        finally:
            os.unlink(data_path)

        self.assertEqual(step.step_ids, ["validate-data-file", "embed-and-upsert-to-pinecone"])
        ingest_fn.assert_called_once_with(data_path)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["records_processed"], 7)

    def test_sync_impl_raises_for_missing_file(self):
        step = FakeStep()
        ingest_fn = Mock(return_value=0)

        with self.assertRaises(FileNotFoundError):
            asyncio.run(_sync_knowledge_base_impl(step, data_path="/tmp/does-not-exist.jsonl", ingest_fn=ingest_fn))


if __name__ == "__main__":
    unittest.main()
