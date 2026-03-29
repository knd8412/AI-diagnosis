import sys
import os
import unittest
import uuid
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from llm_integration.retrieval.chromaClient import PatientMemory

class TestChromaDBIntegration(unittest.TestCase):
    def setUp(self):
        """Initialize PatientMemory with a mocked HttpClient."""
        with patch('llm_integration.retrieval.chromaClient.chromadb.HttpClient'):
            self.memory = PatientMemory()
            self.mock_client = MagicMock()
            self.memory.client = self.mock_client

    @patch('llm_integration.retrieval.chromaClient.embed_query')
    def test_embedding_function_single_string(self, mock_embed):
        """Test embedding function works with single string."""
        ef = self.memory.internal_ef
        mock_embed.return_value = [0.1, 0.1, 0.1]
        result = ef("hello")
        self.assertEqual(len(result), 1)

    @patch('llm_integration.retrieval.chromaClient.embed_query')
    def test_embedding_function_batch(self, mock_embed):
        """Test embedding function works with batch of texts."""
        ef = self.memory.internal_ef
        mock_embed.side_effect = [[0.1, 0.2], [0.2, 0.3]]
        result = ef(["a", "b"])
        self.assertEqual(len(result), 2)

    def test_get_collection(self):
        """Test get_collection creates collection with correct name."""
        mock_col = MagicMock()
        self.mock_client.get_or_create_collection.return_value = mock_col
        
        self.memory._get_collection("abc-123")
        
        call_kwargs = self.mock_client.get_or_create_collection.call_args[1]
        self.assertEqual(call_kwargs['name'], "patient_abc-123")

    @patch('uuid.uuid4')
    def test_add_interaction(self, mock_uuid):
        """Test add_interaction stores data correctly."""
        mock_uuid.return_value = "test-uuid"
        mock_col = MagicMock()
        self.mock_client.get_or_create_collection.return_value = mock_col
        
        self.memory.add_interaction("p1", "doctor", "Patient stable")
        
        call_kwargs = mock_col.add.call_args[1]
        self.assertEqual(call_kwargs['documents'], ["Patient stable"])
        self.assertEqual(call_kwargs['metadatas'][0]['role'], "doctor")

    def test_search_history(self):
        """Test search_history retrieves and joins results."""
        mock_col = MagicMock()
        mock_col.query.return_value = {'documents': [["Note 1", "Note 2"]]}
        self.mock_client.get_or_create_collection.return_value = mock_col
        
        result = self.memory.search_history("p1", "symptoms")
        self.assertEqual(result, "Note 1\nNote 2")

    def test_search_history_empty(self):
        """Test search_history returns empty string when no results."""
        mock_col = MagicMock()
        mock_col.query.return_value = {'documents': [[]]}
        self.mock_client.get_or_create_collection.return_value = mock_col
        
        result = self.memory.search_history("p1", "query")
        self.assertEqual(result, "")

    def test_search_result_joining(self):
        """Test multiple documents are joined by newlines."""
        mock_col = MagicMock()
        mock_col.query.return_value = {'documents': [["Note 1", "Note 2", "Note 3"]]}
        self.mock_client.get_or_create_collection.return_value = mock_col
        
        result = self.memory.search_history("p1", "query")
        self.assertEqual(result, "Note 1\nNote 2\nNote 3")

    def test_search_error_propagation(self):
        """Test search errors are propagated to caller."""
        mock_col = MagicMock()
        mock_col.query.side_effect = Exception("Service Down")
        self.mock_client.get_or_create_collection.return_value = mock_col
        
        with self.assertRaises(Exception) as cm:
            self.memory.search_history("p1", "query")
        self.assertIn("Service Down", str(cm.exception))

if __name__ == '__main__':
    unittest.main()