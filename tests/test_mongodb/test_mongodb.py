import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from bson import ObjectId

import importlib.util, os

def make_client():
    spec = importlib.util.spec_from_file_location(
        "mongo_service",
        os.path.join(os.path.dirname(__file__), "../../mongo_service/service.py")
    )
    assert spec is not None and spec.loader is not None
    service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service)
    setattr(service, "db", MagicMock())
    return TestClient(service.app), service


class TestResults(unittest.TestCase):
    def test_save_result_returns_id(self):
        client, service = make_client()
        mock_insert = MagicMock()
        mock_insert.inserted_id = ObjectId("6" * 24)
        service.db["analysis_results"].insert_one = AsyncMock(return_value=mock_insert)

        response = client.post("/results", json={"pathologies": {"Pneumonia": 0.87}})
        self.assertEqual(response.status_code, 200)
        self.assertIn("id", response.json())

    def test_get_result_not_found(self):
        client, service = make_client()
        service.db["analysis_results"].find_one = AsyncMock(return_value=None)

        response = client.get(f"/results/{'6' * 24}")
        self.assertEqual(response.status_code, 404)

    def test_get_pathologies_only(self):
        client, service = make_client()
        mock_doc = {"_id": ObjectId("6" * 24), "pathologies": {"Pneumonia": 0.9}}
        service.db["analysis_results"].find_one = AsyncMock(return_value=mock_doc)

        response = client.get(f"/results/{'6' * 24}/pathologies")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Pneumonia", response.json())


class TestChatLogs(unittest.TestCase):
    def test_save_chatlog_upserts(self):
        client, service = make_client()
        service.db["chat_logs"].update_one = AsyncMock()

        response = client.post("/chatlogs", json={"session_id": "sess_1", "messages": ["hello"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_get_chatlog_not_found(self):
        client, service = make_client()
        service.db["chat_logs"].find_one = AsyncMock(return_value=None)

        response = client.get("/chatlogs/nonexistent")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
