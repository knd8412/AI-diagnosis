"""
Test Suite for Embedding Service
Tests the Docker embedding service independently without requiring other services
"""

import os
import sys
import unittest
import requests

# 1. PATH SETUP
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class TestEmbeddingService(unittest.TestCase):
    """Test the embedding service API endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Port 5001 per your docker-compose.yml
        cls.base_url = os.environ.get("EMBEDDING_SERVICE_URL", "http://localhost:5001")
        
        # Check if service is online
        try:
            response = requests.get(f"{cls.base_url}/health", timeout=5)
            cls.service_online = response.status_code == 200
        except Exception as e:
            cls.service_online = False
            print(f"\n⚠️  Warning: Embedding service not reachable at {cls.base_url}")
            print(f"   Error: {e}")
            print(f"   Make sure Docker is running: docker-compose up -d embedding-service\n")

    def test_1_health_check(self):
        """Test that the health endpoint is accessible and returns correct status."""
        if not self.service_online:
            self.skipTest("Docker embedding service is offline")
        
        response = requests.get(f"{self.base_url}/health")
        
        # Check status code
        self.assertEqual(response.status_code, 200, "Health check should return 200")
        
        # Check response content
        data = response.json()
        self.assertIn("status", data, "Health response should contain 'status'")
        self.assertEqual(data["status"], "healthy", "Status should be 'healthy'")
        self.assertIn("model", data, "Health response should contain 'model'")
        self.assertIn("dimension", data, "Health response should contain 'dimension'")
        
        # Verify expected model and dimension
        self.assertEqual(data["dimension"], 1024, "Dimension should be 1024 for Mistral")
        
        print(f"\n✓ Health check passed: {data}")

    def test_2_single_embedding(self):
        """Test creating a single embedding."""
        if not self.service_online:
            self.skipTest("Docker embedding service is offline")
        
        # Test payload
        payload = {"text": "Patient has a persistent cough and fever."}
        
        # Make request
        response = requests.post(f"{self.base_url}/embed", json=payload, timeout=30)
        
        # Check status code
        self.assertEqual(response.status_code, 200, "Embed endpoint should return 200")
        
        # Check response content
        data = response.json()
        self.assertIn("embedding", data, "Response should contain 'embedding'")
        self.assertIn("dimension", data, "Response should contain 'dimension'")
        self.assertIn("model", data, "Response should contain 'model'")
        
        # Verify embedding
        embedding = data["embedding"]
        self.assertIsInstance(embedding, list, "Embedding should be a list")
        self.assertEqual(len(embedding), 1024, "Embedding should have 1024 dimensions")
        
        # Check that embedding contains numbers
        self.assertTrue(all(isinstance(x, (int, float)) for x in embedding[:10]), 
                       "Embedding should contain numeric values")
        
        print(f"\n✓ Single embedding test passed")
        print(f"  Text: {payload['text']}")
        print(f"  Dimension: {data['dimension']}")
        print(f"  First 5 values: {embedding[:5]}")

    def test_3_batch_embeddings(self):
        """Test creating multiple embeddings at once."""
        if not self.service_online:
            self.skipTest("Docker embedding service is offline")
        
        # Test payload with multiple texts
        texts = [
            "Patient has cough and fever",
            "Headache and nausea symptoms",
            "Chest pain and difficulty breathing"
        ]
        payload = {"texts": texts}
        
        # Make request
        response = requests.post(f"{self.base_url}/embed/batch", json=payload, timeout=30)
        
        # Check status code
        self.assertEqual(response.status_code, 200, "Batch embed should return 200")
        
        # Check response content
        data = response.json()
        self.assertIn("embeddings", data, "Response should contain 'embeddings'")
        self.assertIn("count", data, "Response should contain 'count'")
        self.assertIn("dimension", data, "Response should contain 'dimension'")
        
        # Verify embeddings
        embeddings = data["embeddings"]
        self.assertEqual(len(embeddings), len(texts), f"Should return {len(texts)} embeddings")
        self.assertEqual(data["count"], len(texts), f"Count should be {len(texts)}")
        
        # Check each embedding
        for i, embedding in enumerate(embeddings):
            self.assertIsInstance(embedding, list, f"Embedding {i} should be a list")
            self.assertEqual(len(embedding), 1024, f"Embedding {i} should have 1024 dimensions")
        
        print(f"\n✓ Batch embedding test passed")
        print(f"  Texts processed: {len(texts)}")
        print(f"  Embeddings returned: {data['count']}")

    def test_4_error_handling_empty_text(self):
        """Test that empty text is handled correctly."""
        if not self.service_online:
            self.skipTest("Docker embedding service is offline")
        
        # Test with empty text
        payload = {"text": ""}
        response = requests.post(f"{self.base_url}/embed", json=payload, timeout=10)
        
        # Should return 400 error
        self.assertEqual(response.status_code, 400, "Empty text should return 400")
        
        data = response.json()
        self.assertIn("error", data, "Error response should contain 'error' field")
        
        print(f"\n✓ Empty text error handling passed")

    def test_5_error_handling_missing_field(self):
        """Test that missing 'text' field is handled correctly."""
        if not self.service_online:
            self.skipTest("Docker embedding service is offline")
        
        # Test with missing 'text' field
        payload = {"wrong_field": "some value"}
        response = requests.post(f"{self.base_url}/embed", json=payload, timeout=10)
        
        # Should return 400 error
        self.assertEqual(response.status_code, 400, "Missing field should return 400")
        
        data = response.json()
        self.assertIn("error", data, "Error response should contain 'error' field")
        
        print(f"\n✓ Missing field error handling passed")

    def test_6_performance_timing(self):
        """Test the response time of the embedding service."""
        if not self.service_online:
            self.skipTest("Docker embedding service is offline")
        
        import time
        
        payload = {"text": "Test performance of embedding generation"}
        
        start_time = time.time()
        response = requests.post(f"{self.base_url}/embed", json=payload, timeout=30)
        elapsed_time = time.time() - start_time
        
        self.assertEqual(response.status_code, 200, "Request should succeed")
        
        # Check that response is reasonably fast (under 10 seconds)
        self.assertLess(elapsed_time, 10.0, "Embedding should be generated in under 10 seconds")
        
        print(f"\n✓ Performance test passed")
        print(f"  Response time: {elapsed_time:.2f} seconds")


class TestEmbeddingServiceIntegration(unittest.TestCase):
    """
    Integration tests that require other services to be running.
    These tests will be skipped if services are not available.
    """
    
    @classmethod
    def setUpClass(cls):
        """Check if ChromaDB and other services are available."""
        cls.chromadb_available = False
        try:
            response = requests.get("http://localhost:8002/api/v1/heartbeat", timeout=2)
            cls.chromadb_available = response.status_code == 200
        except:
            pass
    
    def test_integration_with_chromadb(self):
        """
        Test integration with ChromaDB (requires ChromaDB to be running).
        This test will be skipped if ChromaDB is not available.
        """
        if not self.chromadb_available:
            self.skipTest("ChromaDB service is not running. Start with: docker-compose up -d chromadb")
        
        # Import only if we're actually running the test
        try:
            from llm_integration.chains.rag_chains import DiagnosisRAG
            
            # Try to initialize RAG
            rag = DiagnosisRAG()
            self.assertIsNotNone(rag, "DiagnosisRAG should initialize successfully")
            
            print("\n✓ ChromaDB integration test passed")
        except Exception as e:
            self.fail(f"Integration test failed: {e}")


def run_tests():
    """Run all tests with verbose output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add embedding service tests
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingService))
    
    # Add integration tests (will be skipped if services not available)
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingServiceIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Check if embedding service is running
    try:
        response = requests.get("http://localhost:5001/health", timeout=2)
        if response.status_code == 200:
            print("\n✓ Embedding service is running")
            print(f"  Response: {response.json()}\n")
        else:
            print("\n⚠️  Embedding service responded with unexpected status")
    except Exception as e:
        print("\n❌ Embedding service is not running!")
        print("   Start it with: docker-compose up -d embedding-service")
        print(f"   Error: {e}\n")
    
    # Run tests
    success = run_tests()
    sys.exit(0 if success else 1)
