"""
Test Script for Docker Embedding Service
Run this to verify your Docker service is working correctly
"""

import os
import sys
import requests
import json
import time

# Add the parent directory to sys.path so we can import the assistant
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from medical_diagnosis_assistant import MedicalDiagnosisAssistant
except ImportError:
    print("Error: Could not import MedicalDiagnosisAssistant.")
    print(f"Current sys.path includes: {sys.path[0]}")


def test_health_check(base_url="http://localhost:5000"):
    """Test the health check endpoint."""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Health check PASSED")
            print(f"  Status: {data.get('status')}")
            print(f"  Model: {data.get('model')}")
            print(f"  Dimension: {data.get('dimension')}")
            return True
        else:
            print(f"✗ Health check FAILED: Status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Health check FAILED: {e}")
        print("\nTroubleshooting:")
        print("1. Is Docker Desktop running?")
        print("2. Is the service started? Run: docker-compose up -d")
        print("3. Check logs: docker-compose logs")
        return False


def test_single_embedding(base_url="http://localhost:5000"):
    """Test creating a single embedding."""
    print("\n" + "="*60)
    print("TEST 2: Single Embedding")
    print("="*60)
    
    test_text = "I have a fever, cough, and difficulty breathing"
    print(f"Input text: {test_text}")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/embed",
            json={"text": test_text},
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            embedding = data.get('embedding', [])
            
            print(f"✓ Embedding creation PASSED")
            print(f"  Dimension: {data.get('dimension')}")
            print(f"  Model: {data.get('model')}")
            print(f"  Time taken: {elapsed:.2f} seconds")
            print(f"  First 5 values: {embedding[:5]}")
            return True
        else:
            print(f"✗ Embedding creation FAILED: Status code {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    
    except requests.exceptions.RequestException as e:
        print(f"✗ Embedding creation FAILED: {e}")
        return False


def test_batch_embeddings(base_url="http://localhost:5000"):
    """Test creating batch embeddings."""
    print("\n" + "="*60)
    print("TEST 3: Batch Embeddings")
    print("="*60)
    
    test_texts = [
        "fever and cough",
        "headache and nausea",
        "chest pain and shortness of breath"
    ]
    print(f"Input: {len(test_texts)} texts")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/embed/batch",
            json={"texts": test_texts},
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✓ Batch embedding PASSED")
            print(f"  Count: {data.get('count')}")
            print(f"  Dimension: {data.get('dimension')}")
            print(f"  Model: {data.get('model')}")
            print(f"  Time taken: {elapsed:.2f} seconds")
            print(f"  Average time per text: {elapsed/len(test_texts):.2f} seconds")
            return True
        else:
            print(f"✗ Batch embedding FAILED: Status code {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    
    except requests.exceptions.RequestException as e:
        print(f"✗ Batch embedding FAILED: {e}")
        return False


def test_error_handling(base_url="http://localhost:5000"):
    """Test error handling."""
    print("\n" + "="*60)
    print("TEST 4: Error Handling")
    print("="*60)
    
    all_passed = True # Track success
    
    # Test with empty text
    print("\nTesting with empty text...")
    try:
        response = requests.post(
            f"{base_url}/embed",
            json={"text": ""},
            timeout=10
        )
        
        if response.status_code == 400:
            print("✓ Empty text handling PASSED (correctly rejected)")
        else:
            print(f"✗ Empty text handling FAILED: Expected 400, got {response.status_code}")
            all_passed = False
    except Exception as e:
        print(f"✗ Empty text test FAILED: {e}")
        all_passed = False
    
    # Test with missing field
    print("\nTesting with missing 'text' field...")
    try:
        response = requests.post(
            f"{base_url}/embed",
            json={"wrong_field": "test"},
            timeout=10
        )
        
        if response.status_code == 400:
            print("✓ Missing field handling PASSED (correctly rejected)")
        else:
            print(f"✗ Missing field handling FAILED: Expected 400, got {response.status_code}")
            all_passed = False
    except Exception as e:
        print(f"✗ Missing field test FAILED: {e}")
        all_passed = False

    return all_passed


def test_integration_with_assistant():
    """Test integration with MedicalDiagnosisAssistant."""
    print("\n" + "="*60)
    print("TEST 5: Integration with Assistant")
    print("="*60)
    
    try:
        assistant = MedicalDiagnosisAssistant(use_docker_embeddings=True)        
        print("Assistant initialized from tests/ subfolder")

        print("Testing embedding creation...")
        test_text = "fever and cough"
        embedding = assistant.create_embedding(test_text)
        
        if embedding and len(embedding) == 1024: 
            print(f"✓ Integration test PASSED")
            print(f"  Created embedding with {len(embedding)} dimensions")
            return True
        else:
            print(f"✗ Integration test FAILED: Unexpected dimension {len(embedding) if embedding else 'None'} (Expected 1024)")
            return False
    
    except Exception as e:
        print(f"✗ Integration test FAILED: {e}")
        return False

def test_clinical_metadata_extraction():
    """Test the LLM-based metadata extraction feature."""
    print("\n" + "="*60)
    print("TEST 6: Clinical Metadata Extraction")
    print("="*60)
    
    try:
        assistant = MedicalDiagnosisAssistant(use_docker_embeddings=True)
        test_query = "I've had a bad cough for 3 days and a high fever"
        
        # Call the new method
        metadata = assistant.extract_clinical_metadata(test_query)
        
        # Verify the structure of the returned JSON
        required_keys = ['symptoms', 'duration']
        if all(key in metadata for key in required_keys):
            print("✓ Metadata extraction PASSED")
            print(f"  Extracted: {metadata}")
            return True
        else:
            print(f"✗ Metadata extraction FAILED: Missing keys. Got {metadata.keys()}")
            return False
            
    except Exception as e:
        print(f"✗ Metadata extraction FAILED: {e}")
        return False

def run_all_tests(base_url="http://localhost:5000"):
    """Run all tests."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "DOCKER SERVICE TESTS" + " "*23 + "║")
    print("╚" + "="*58 + "╝")
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health_check(base_url)))
    
    # Only continue if health check passes
    if results[0][1]:
        results.append(("Single Embedding", test_single_embedding(base_url)))
        results.append(("Batch Embeddings", test_batch_embeddings(base_url)))
        results.append(("Error Handling", test_error_handling(base_url)))
        results.append(("Integration", test_integration_with_assistant()))
        results.append(("Metadata Extraction", test_clinical_metadata_extraction()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print("-"*60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n ✓✓✓ All tests passed! Your Docker service is working correctly.")
        print("\nYou can now use it in your code:")
        print("  assistant = MedicalDiagnosisAssistant(")
        print("      use_docker_embeddings=True,")
        print('      docker_url="http://localhost:5000"')
        print("  )")
    else:
        print("\n ✗✗✗ Some tests failed. Check the output above for details.")
    
    print("="*60)


if __name__ == "__main__":
    # You can change the URL here if using a different port
    SERVICE_URL = "http://localhost:5000"
    run_all_tests(SERVICE_URL)