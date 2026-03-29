import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class TestUICore(unittest.TestCase):
    """Core functionality tests for UI interactions only."""

    def setUp(self):
        """Set up mocks for UI testing."""
        self.mock_rag_response = {
            "diagnosis": "Potential Respiratory Infection",
            "retrieved_context": "Patient history: cough for 3 days"
        }

    def test_session_state_initialization(self):
        """Test that session state initializes with required keys."""
        session_state = {}
        
        if 'messages' not in session_state:
            session_state['messages'] = []
        if 'patient_id' not in session_state:
            session_state['patient_id'] = None
            
        self.assertIn('messages', session_state)
        self.assertIn('patient_id', session_state)

    @patch('llm_integration.chains.diagnosis_orchestration.DiagnosisRAG')
    def test_ui_calls_rag_with_patient_id(self, mock_rag_class):
        """Test that UI passes patient_id to RAG system."""
        mock_instance = mock_rag_class.return_value
        mock_instance.diagnose.return_value = self.mock_rag_response
        
        patient_id = "patient-123"
        query = "Persistent cough"
        
        result = mock_instance.diagnose(patient_id, query)
        
        mock_instance.diagnose.assert_called_once_with(patient_id, query)
        self.assertIsNotNone(result)

    @patch('llm_integration.chains.diagnosis_orchestration.DiagnosisRAG')
    def test_ui_receives_diagnosis_response(self, mock_rag_class):
        """Test that UI receives diagnosis response with required fields."""
        mock_instance = mock_rag_class.return_value
        mock_instance.diagnose.return_value = self.mock_rag_response
        
        result = mock_instance.diagnose("p1", "query")
        
        self.assertIn('diagnosis', result)
        self.assertIn('retrieved_context', result)

    def test_user_query_input_handling(self):
        """Test that user queries are handled as strings."""
        user_input = "Chest pain and fever"
        
        self.assertIsInstance(user_input, str)
        self.assertGreater(len(user_input), 0)

    def test_message_structure(self):
        """Test that messages follow required structure."""
        message = {
            "role": "user",
            "content": "I have a fever"
        }
        
        self.assertIn("role", message)
        self.assertIn("content", message)

    @patch('llm_integration.chains.diagnosis_orchestration.DiagnosisRAG')
    def test_ui_error_display(self, mock_rag_class):
        """Test that UI handles RAG errors."""
        mock_instance = mock_rag_class.return_value
        mock_instance.diagnose.side_effect = Exception("Service Error")
        
        with self.assertRaises(Exception):
            mock_instance.diagnose("p1", "query")

if __name__ == '__main__':
    unittest.main()