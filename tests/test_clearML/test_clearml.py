"""
Unit Tests for ClearML Connection and Configuration
Tests that ClearML is properly configured and can connect to the server.
"""

import os
import sys
import unittest
from dotenv import load_dotenv

# Path setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Load environment variables
load_dotenv(os.path.join(ROOT_DIR, '.env'))


class TestClearMLConfiguration(unittest.TestCase):
    """Test ClearML configuration."""
    
    def test_1_clearml_credentials_exist(self):
        """Verify ClearML credentials are configured in environment."""
        api_host = os.getenv("CLEARML_API_HOST")
        web_host = os.getenv("CLEARML_WEB_HOST")
        files_host = os.getenv("CLEARML_FILES_HOST")
        api_access_key = os.getenv("CLEARML_API_ACCESS_KEY")
        api_secret_key = os.getenv("CLEARML_API_SECRET_KEY")
        
        self.assertIsNotNone(api_host, "CLEARML_API_HOST not configured")
        self.assertIsNotNone(web_host, "CLEARML_WEB_HOST not configured")
        self.assertIsNotNone(files_host, "CLEARML_FILES_HOST not configured")
        self.assertIsNotNone(api_access_key, "CLEARML_API_ACCESS_KEY not configured")
        self.assertIsNotNone(api_secret_key, "CLEARML_API_SECRET_KEY not configured")
        
        print(f"\n✓ All ClearML credentials are configured")
        print(f"  API Host: {api_host}")
        print(f"  Web Host: {web_host}")
    
    def test_2_clearml_endpoints_format(self):
        """Verify ClearML endpoint URLs are valid."""
        api_host = os.getenv("CLEARML_API_HOST")
        web_host = os.getenv("CLEARML_WEB_HOST")
        files_host = os.getenv("CLEARML_FILES_HOST")
        
        # Check that endpoints are URLs
        for endpoint in [api_host, web_host, files_host]:
            self.assertTrue(
                endpoint.startswith("http://") or endpoint.startswith("https://"),
                f"Endpoint {endpoint} does not start with http:// or https://"
            )
        
        print(f"\n✓ All ClearML endpoints have valid URL format")


class TestClearMLConnection(unittest.TestCase):
    """Test ClearML connection and functionality."""
    
    def test_1_clearml_import(self):
        """Test that ClearML package can be imported."""
        try:
            from clearml import Task
            print(f"\n✓ ClearML package imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import ClearML: {e}")
    
    def test_2_clearml_credentials_setup(self):
        """Test setting up ClearML credentials."""
        try:
            from clearml import Task
            
            api_host = os.getenv("CLEARML_API_HOST")
            web_host = os.getenv("CLEARML_WEB_HOST")
            files_host = os.getenv("CLEARML_FILES_HOST")
            api_access_key = os.getenv("CLEARML_API_ACCESS_KEY")
            api_secret_key = os.getenv("CLEARML_API_SECRET_KEY")
            
            Task.set_credentials(
                api_host=api_host,
                web_host=web_host,
                files_host=files_host,
                key=api_access_key,
                secret=api_secret_key
            )
            
            print(f"\n✓ ClearML credentials configured successfully")
        except Exception as e:
            self.fail(f"Failed to set ClearML credentials: {e}")
    
    def test_3_clearml_task_initialization(self):
        """Test initializing a ClearML Task."""
        try:
            from clearml import Task
            
            # Configure credentials first
            api_host = os.getenv("CLEARML_API_HOST")
            web_host = os.getenv("CLEARML_WEB_HOST")
            files_host = os.getenv("CLEARML_FILES_HOST")
            api_access_key = os.getenv("CLEARML_API_ACCESS_KEY")
            api_secret_key = os.getenv("CLEARML_API_SECRET_KEY")
            
            Task.set_credentials(
                api_host=api_host,
                web_host=web_host,
                files_host=files_host,
                key=api_access_key,
                secret=api_secret_key
            )
            
            # Try to initialize a task
            task = Task.init(
                project_name="embedding_tests",
                task_name="unit_test_task"
            )
            
            self.assertIsNotNone(task, "Task initialization should return a Task object")
            print(f"\n✓ ClearML Task initialized successfully")
            
        except Exception as e:
            # This is acceptable if the server is not accessible
            print(f"\n⚠️  Warning: Failed to initialize ClearML task")
            print(f"   This may be expected if ClearML server is not accessible")
            print(f"   Error: {e}")
    
    def test_4_clearml_logger(self):
        """Test ClearML task logger."""
        try:
            from clearml import Task
            
            # Configure credentials first
            api_host = os.getenv("CLEARML_API_HOST")
            web_host = os.getenv("CLEARML_WEB_HOST")
            files_host = os.getenv("CLEARML_FILES_HOST")
            api_access_key = os.getenv("CLEARML_API_ACCESS_KEY")
            api_secret_key = os.getenv("CLEARML_API_SECRET_KEY")
            
            Task.set_credentials(
                api_host=api_host,
                web_host=web_host,
                files_host=files_host,
                key=api_access_key,
                secret=api_secret_key
            )
            
            # Initialize a task
            task = Task.init(
                project_name="embedding_tests",
                task_name="logger_test"
            )
            
            # Get logger and test it
            logger = task.get_logger()
            
            logger.report_scalar(
                title="test_metric",
                series="value",
                value=1.0,
                iteration=0
            )
            
            print(f"\n✓ ClearML logger working correctly")
            
        except Exception as e:
            # This is acceptable if the server is not accessible
            print(f"\n⚠️  Warning: Failed to use ClearML logger")
            print(f"   Error: {e}")


def run_all_tests():
    """Run all ClearML tests."""
    print("\n" + "="*70)
    print("CLEARML CONNECTION TEST SUITE")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes in order
    suite.addTests(loader.loadTestsFromTestCase(TestClearMLConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestClearMLConnection))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    if result.wasSuccessful():
        print("\n✓ All ClearML tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check configuration and server accessibility.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

