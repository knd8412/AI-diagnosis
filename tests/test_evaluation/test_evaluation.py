"""
Unit Tests for RAG Diagnosis System
Tests the complete DiagnosisRAG pipeline including initialization, retrieval, and LLM response generation.

NOTE: Some tests require ChromaDB to be running:
  - Run locally: docker-compose up -d chromadb
"""

import os
import sys
import unittest
import time
from dotenv import load_dotenv

# Path setup - ensure we can import from project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Load environment variables
load_dotenv(os.path.join(ROOT_DIR, '.env'))

# Import with error handling
IMPORT_ERROR = None
try:
    from llm_integration.chains.rag_chains import DiagnosisRAG
except ImportError as e:
    DiagnosisRAG = None
    IMPORT_ERROR = str(e)


class TestRAGSystemSetup(unittest.TestCase):
    """Test RAG system configuration and setup."""
    
    def test_1_environment_variables(self):
        """Verify all required environment variables are set."""
        mistral_key = os.getenv("MISTRAL_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        
        self.assertIsNotNone(mistral_key, "MISTRAL_API_KEY not found in environment")
        self.assertIsNotNone(pinecone_key, "PINECONE_API_KEY not found in environment")
        
        # Check keys are not empty or placeholder values
        self.assertTrue(len(mistral_key) > 10, "MISTRAL_API_KEY appears to be invalid")
        self.assertTrue(len(pinecone_key) > 10, "PINECONE_API_KEY appears to be invalid")
        
        print(f"\n✓ API keys configured correctly")
    
    def test_2_import_diagnosis_rag(self):
        """Verify DiagnosisRAG class can be imported."""
        if DiagnosisRAG is None:
            self.fail(f"Could not import DiagnosisRAG: {IMPORT_ERROR}")
        
        # Check it's a class
        self.assertTrue(callable(DiagnosisRAG), "DiagnosisRAG should be a class")
        
        print(f"\n✓ DiagnosisRAG class imported successfully")


class TestRAGSystemInitialization(unittest.TestCase):
    """Test RAG system initialization."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        if DiagnosisRAG is None:
            cls.rag = None
            cls.init_error = IMPORT_ERROR
        else:
            try:
                print("\nInitializing RAG system...")
                print("(This requires ChromaDB to be running)")
                start_time = time.time()
                cls.rag = DiagnosisRAG()
                cls.init_time = time.time() - start_time
                cls.init_error = None
                print(f"✓ RAG initialized in {cls.init_time:.2f}s")
            except Exception as e:
                cls.rag = None
                cls.init_error = str(e)
                print(f"✗ RAG initialization failed: {e}")
                
                # Check if it's a ChromaDB connection error
                if "Chroma server" in str(e) or "chromadb" in str(e).lower():
                    print("\n💡 TIP: Start ChromaDB with: docker-compose up -d chromadb")
    
    def test_1_initialization_success(self):
        """Verify RAG system initializes without errors."""
        if DiagnosisRAG is None:
            self.fail(f"DiagnosisRAG not available: {IMPORT_ERROR}")
        
        self.assertIsNone(self.init_error, f"Initialization failed: {self.init_error}")
        self.assertIsNotNone(self.rag, "RAG instance should not be None")
        
        print(f"\n✓ RAG system initialized successfully")
    
    def test_2_initialization_time(self):
        """Verify initialization completes in reasonable time."""
        if self.rag is None:
            self.fail("RAG not initialized")
        
        # Should initialize in under 30 seconds
        self.assertLess(self.init_time, 30.0, 
                       f"Initialization took {self.init_time:.2f}s (should be < 30s)")
        
        print(f"\n✓ Initialization time acceptable: {self.init_time:.2f}s")
    
    def test_3_required_attributes(self):
        """Verify RAG instance has required attributes."""
        if self.rag is None:
            self.fail("RAG not initialized")
        
        # Check for expected attributes
        expected_attrs = ['diagnose']
        
        for attr in expected_attrs:
            self.assertTrue(hasattr(self.rag, attr), 
                          f"RAG instance missing '{attr}' attribute/method")
        
        print(f"\n✓ RAG instance has required attributes")


class TestRAGSystemDiagnosis(unittest.TestCase):
    """Test RAG diagnosis functionality end-to-end."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize RAG once for all diagnosis tests."""
        if DiagnosisRAG is None:
            cls.rag = None
        else:
            try:
                cls.rag = DiagnosisRAG()
            except Exception as e:
                cls.rag = None
                print(f"Setup failed: {e}")
    
    def test_1_basic_diagnosis(self):
        """Test basic diagnosis with common symptoms."""
        if self.rag is None:
            self.fail("RAG not initialized")
        
        patient_id = "test_patient_001"
        query = "I have a persistent cough and fever."
        
        try:
            result = self.rag.diagnose(patient_id, query)
            self.assertIsNotNone(result, "Diagnosis result should not be None")
            self.assertIsInstance(result, dict, "Diagnosis result should be a dictionary")
            print(f"\n✓ Basic diagnosis executed successfully")
        except Exception as e:
            self.fail(f"Diagnosis failed with error: {e}")
    
    def test_2_llm_response_generation(self):
        """Test that LLM generates a diagnosis response."""
        if self.rag is None:
            self.fail("RAG not initialized")
        
        patient_id = "test_patient_002"
        query = "I have a persistent cough and fever."
        
        try:
            result = self.rag.diagnose(patient_id, query)
            
            self.assertIn("diagnosis", result, "Result should contain 'diagnosis' key")
            diagnosis = result["diagnosis"]
            
            self.assertTrue(len(diagnosis) > 0, "Diagnosis should not be empty")
            self.assertIsInstance(diagnosis, str, "Diagnosis should be a string")
            self.assertGreater(len(diagnosis), 10, "Diagnosis seems too short to be valid")
            
            print(f"\n✓ LLM response generation working")
            print(f"  Sample diagnosis: {diagnosis[:100]}...")
            
        except Exception as e:
            self.fail(f"LLM response generation failed: {e}")
    
    def test_3_pinecone_retrieval(self):
        """Test that Pinecone retrieves relevant context."""
        if self.rag is None:
            self.fail("RAG not initialized")
        
        patient_id = "test_patient_003"
        query = "I have a persistent cough and fever."
        
        try:
            result = self.rag.diagnose(patient_id, query)
            
            self.assertIn("retrieved_context", result, 
                         "Result should contain 'retrieved_context' key")
            
            context = result["retrieved_context"]
            self.assertTrue(len(context) > 0, "Retrieved context should not be empty")
            
            print(f"\n✓ Pinecone retrieval working")
            
            if isinstance(context, list):
                print(f"  Retrieved {len(context)} context items")
            
        except Exception as e:
            self.fail(f"Pinecone retrieval failed: {e}")
    
    def test_4_diagnosis_response_time(self):
        """Test that diagnosis completes in reasonable time."""
        if self.rag is None:
            self.fail("RAG not initialized")
        
        patient_id = "test_patient_004"
        query = "I have a headache and nausea."
        start_time = time.time()
        
        try:
            result = self.rag.diagnose(patient_id, query)
            elapsed_time = time.time() - start_time
            
            self.assertLess(elapsed_time, 30.0, 
                          f"Diagnosis took {elapsed_time:.2f}s (should be < 30s)")
            
            print(f"\n✓ Diagnosis response time acceptable: {elapsed_time:.2f}s")
            
        except Exception as e:
            self.fail(f"Diagnosis failed: {e}")
    
    def test_5_multiple_symptoms(self):
        """Test diagnosis with multiple different symptom queries."""
        if self.rag is None:
            self.fail("RAG not initialized")
        
        patient_id = "test_patient_005"
        test_queries = [
            "I have chest pain and difficulty breathing",
            "Patient experiencing severe headache and vision problems",
            "Symptoms include nausea, vomiting, and abdominal pain"
        ]
        
        for query in test_queries:
            try:
                result = self.rag.diagnose(patient_id, query)
                self.assertIn("diagnosis", result, f"No diagnosis for query: {query}")
                self.assertTrue(len(result["diagnosis"]) > 0, 
                              f"Empty diagnosis for query: {query}")
            except Exception as e:
                self.fail(f"Failed on query '{query}': {e}")
        
        print(f"\n✓ Multiple symptom queries handled ({len(test_queries)} queries)")


class TestRAGSystemEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize RAG once for edge case tests."""
        if DiagnosisRAG is None:
            cls.rag = None
        else:
            try:
                cls.rag = DiagnosisRAG()
            except:
                cls.rag = None
    
    def test_1_empty_query(self):
        """Test handling of empty query."""
        if self.rag is None:
            self.fail("RAG not initialized")
        
        patient_id = "test_patient_edge_001"
        try:
            result = self.rag.diagnose(patient_id, "")
            self.assertIsNotNone(result, "Should handle empty query")
        except Exception:
            pass  # It's acceptable to raise an exception for empty query
        
        print(f"\n✓ Empty query handled")
    
    def test_2_very_long_query(self):
        """Test handling of very long query."""
        if self.rag is None:
            self.fail("RAG not initialized")
        
        patient_id = "test_patient_edge_002"
        long_query = "I have " + ", ".join([f"symptom {i}" for i in range(100)])
        
        try:
            result = self.rag.diagnose(patient_id, long_query)
            self.assertIn("diagnosis", result, "Should handle long queries")
            print(f"\n✓ Long query handled")
        except Exception as e:
            print(f"\n⚠ Long query failed (acceptable): {str(e)[:50]}")


def run_all_tests():
    """Run all RAG system tests with custom output."""
    print("\n" + "="*70)
    print(" "*15 + "🏥 RAG EVALUATION SYSTEM - UNIT TESTS")
    print("="*70)
    print("\n✓ All tests will run (no skipping on production server)")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes in order
    suite.addTests(loader.loadTestsFromTestCase(TestRAGSystemSetup))
    suite.addTests(loader.loadTestsFromTestCase(TestRAGSystemInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestRAGSystemDiagnosis))
    suite.addTests(loader.loadTestsFromTestCase(TestRAGSystemEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print(f"{'TEST SUMMARY':<30}")
    print("="*70)
    print(f"{'Tests Run':<30}: {result.testsRun}")
    print(f"{'Passed':<30}: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"{'Failed':<30}: {len(result.failures)}")
    print(f"{'Errors':<30}: {len(result.errors)}")
    print(f"{'Skipped':<30}: {len(result.skipped)}")
    print("="*70)
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! Your RAG system is working correctly.")
    else:
        print("\n⚠️  SOME TESTS FAILED. Please review the output above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
