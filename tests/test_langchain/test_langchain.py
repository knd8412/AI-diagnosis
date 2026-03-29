import unittest
from unittest.mock import Mock, patch
import json


class TestPromptTemplates(unittest.TestCase):

    def test_symptom_extraction_prompt_format(self):
        from llm_integration.prompts.diagnosis_prompts import symptom_extraction_prompt
        result = symptom_extraction_prompt.format(user_input="I have a cough and fever")
        self.assertIn("I have a cough and fever", result)
        self.assertIn("symptoms", result.lower())
        self.assertIn("json", result.lower())

    def test_diagnosis_prompt_format(self):
        from llm_integration.prompts.diagnosis_prompts import diagnosis_prompt
        result = diagnosis_prompt.format(
            context="Medical context here",
            chat_history="Previous chat",
            question="What is wrong with me?"
        )
        self.assertIn("Medical context here", result)
        self.assertIn("Previous chat", result)
        self.assertIn("What is wrong with me?", result)

    def test_explanation_prompt_format(self):
        from llm_integration.prompts.diagnosis_prompts import diagnosis_explanation_prompt
        result = diagnosis_explanation_prompt.format(
            condition="Pneumonia",
            patient_symptoms="cough, fever",
            context="Medical info"
        )
        self.assertIn("Pneumonia", result)
        self.assertIn("cough, fever", result)
        self.assertIn("Medical info", result)


class TestSymptomExtractor(unittest.TestCase):

    @patch('llm_integration.chains.symptom_extraction_chain.get_llm')
    def test_extract_valid_symptoms(self, mock_llm):
        from llm_integration.chains.symptom_extraction_chain import SymptomExtractor
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "symptoms": ["cough", "shortness of breath"],
            "duration": "2 weeks",
            "severity": "moderate",
            "patient_age": 45,
            "patient_gender": "male"
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        extractor = SymptomExtractor()
        result = extractor.extract("45 year old male with cough and breathing difficulty for 2 weeks")
        self.assertEqual(result["symptoms"], ["cough", "shortness of breath"])
        self.assertEqual(result["duration"], "2 weeks")
        self.assertEqual(result["patient_age"], 45)

    @patch('llm_integration.chains.symptom_extraction_chain.get_llm')
    def test_extract_with_markdown_json(self, mock_llm):
        from llm_integration.chains.symptom_extraction_chain import SymptomExtractor
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = '```json\n{"symptoms": ["fever", "cough"]}\n```'
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        extractor = SymptomExtractor()
        result = extractor.extract("I have fever and cough")
        self.assertEqual(result["symptoms"], ["fever", "cough"])
        self.assertNotIn("error", result)

    @patch('llm_integration.chains.symptom_extraction_chain.get_llm')
    def test_extract_invalid_json(self, mock_llm):
        from llm_integration.chains.symptom_extraction_chain import SymptomExtractor
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = "This is not valid JSON"
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        extractor = SymptomExtractor()
        result = extractor.extract("I feel sick")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Failed to parse JSON")


class TestDiagnosisExplainer(unittest.TestCase):

    @patch('llm_integration.chains.explanation_chain.get_llm')
    def test_explain_condition(self, mock_llm):
        from llm_integration.chains.explanation_chain import DiagnosisExplainer
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = "Pneumonia matches symptoms because..."
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        explainer = DiagnosisExplainer()
        result = explainer.explain(
            condition="Pneumonia",
            patient_symptoms="cough, fever, shortness of breath",
            context="Pneumonia clinical presentation..."
        )
        self.assertIn("Pneumonia", result)
        self.assertIn("symptoms", str(result).lower())


class TestImageProcessingClient(unittest.TestCase):

    def test_extract_pathology_symptoms(self):
        from llm_integration.service.image_processing_client import extract_pathology_symptoms
        analysis = {
            "confident_results": {
                "Pneumonia": 0.85,
                "Consolidation": 0.72
            }
        }
        result = extract_pathology_symptoms(analysis)
        self.assertIn("Pneumonia", result)
        self.assertIn("85", result)
        self.assertIn("Consolidation", result)

    def test_extract_pathology_symptoms_empty(self):
        from llm_integration.service.image_processing_client import extract_pathology_symptoms
        result = extract_pathology_symptoms({})
        self.assertIn("No analysis result available", result)

    def test_extract_pathology_symptoms_invalid(self):
        from llm_integration.service.image_processing_client import extract_pathology_symptoms
        result = extract_pathology_symptoms({"confident_results": {}})
        self.assertIn("No significant pathologies", result)





class TestQueryEmbedder(unittest.TestCase):

    @patch('llm_integration.retrieval.query_embedder.requests.post')
    def test_embed_query_success(self, mock_post):
        from llm_integration.retrieval.query_embedder import embed_query
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1] * 1024,
            "dimension": 1024
        }
        mock_post.return_value = mock_response

        result = embed_query("cough and fever")
        self.assertEqual(len(result), 1024)
        self.assertTrue(all(isinstance(x, float) for x in result))

    @patch('llm_integration.retrieval.query_embedder.requests.post')
    def test_embed_query_error(self, mock_post):
        from llm_integration.retrieval.query_embedder import embed_query
        mock_post.side_effect = Exception("Service unavailable")

        with self.assertRaises(Exception):
            embed_query("test query")


if __name__ == "__main__":
    unittest.main()
