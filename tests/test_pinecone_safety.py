import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from data_prep.setup_pinecone import ingest_jsonl_to_pinecone


class FakeIndex:
    def __init__(self):
        self.upserts = []

    def upsert(self, vectors, namespace):
        self.upserts.append((vectors, namespace))


class FakePinecone:
    def __init__(self, api_key):
        self.api_key = api_key
        self.created = False
        self.deleted = False
        self.index = FakeIndex()

    def has_index(self, name):
        return True

    def delete_index(self, name):
        self.deleted = True

    def create_index(self, **kwargs):
        self.created = True

    def describe_index(self, name):
        return SimpleNamespace(status={"ready": True})

    def Index(self, name):
        return self.index


class PineconeSafetyTests(unittest.TestCase):
    def test_ingest_does_not_reset_existing_index(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write(json.dumps({"page_content": "shortness of breath", "metadata": {"source": "unit"}}) + "\n")
            data_path = tmp.name

        fake_pc_instances = []

        def fake_pinecone_factory(api_key):
            instance = FakePinecone(api_key)
            fake_pc_instances.append(instance)
            return instance

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embeddings": [[0.1] * 1024]}

        try:
            with patch.dict(os.environ, {"PINECONE_API_KEY": "test-key"}, clear=False):
                with patch("data_prep.setup_pinecone.Pinecone", side_effect=fake_pinecone_factory):
                    with patch("data_prep.setup_pinecone.requests.post", return_value=FakeResponse()):
                        processed = ingest_jsonl_to_pinecone(data_path)
        finally:
            os.unlink(data_path)

        self.assertEqual(processed, 1)
        self.assertEqual(len(fake_pc_instances), 1)
        fake_pc = fake_pc_instances[0]
        self.assertFalse(fake_pc.deleted, "Existing index must not be deleted")
        self.assertFalse(fake_pc.created, "Existing index must not be recreated")
        self.assertEqual(len(fake_pc.index.upserts), 1)


if __name__ == "__main__":
    unittest.main()
