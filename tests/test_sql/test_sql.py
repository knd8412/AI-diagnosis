import unittest
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Path setup to ensure imports work from the project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import models and helpers
# Note: We will monkey-patch the session for testing
from SQLdb.models import Base, Patient, get_patient, save_patient, get_all_patients
import SQLdb.models as models

class TestSQLDatabase(unittest.TestCase):
    """Unit tests for Patient database operations."""

    @classmethod
    def setUpClass(cls):
        """Set up an in-memory database for testing."""
        cls.engine = create_engine("sqlite:///:memory:")
        cls.SessionTest = sessionmaker(bind=cls.engine)
        Base.metadata.create_all(cls.engine)
        
        # Override the SessionLocal in the models module to use our test DB
        models.SessionLocal = cls.SessionTest

    def setUp(self):
        """Clean the database before each test."""
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

    def test_save_and_get_patient(self):
        """Test creating a new patient and retrieving it by ID."""
        patient_id = "P101"
        name = "John Doe"
        age = 30
        gender = "Male"

        # Save the patient
        save_patient(patient_id, name, age, gender)

        # Retrieve the patient
        patient = get_patient(patient_id)

        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, patient_id)
        self.assertEqual(patient.name, name)
        self.assertEqual(patient.age, age)
        self.assertEqual(patient.gender, gender)

    def test_update_existing_patient(self):
        """Test that save_patient updates records if the ID already exists (merge)."""
        patient_id = "P102"
        save_patient(patient_id, "Original Name", 25, "Female")
        
        # Update the name
        save_patient(patient_id, "Updated Name", 26, "Female")
        
        patient = get_patient(patient_id)
        self.assertEqual(patient.name, "Updated Name")
        self.assertEqual(patient.age, 26)

    def test_get_all_patients(self):
        """Test retrieving all patient records."""
        save_patient("P1", "Patient A", 20, "M")
        save_patient("P2", "Patient B", 40, "F")

        patients = get_all_patients()
        
        self.assertEqual(len(patients), 2)
        names = [p.name for p in patients]
        self.assertIn("Patient A", names)
        self.assertIn("Patient B", names)

    def test_get_nonexistent_patient(self):
        """Test that get_patient returns None for missing IDs."""
        patient = get_patient("NON_EXISTENT")
        self.assertIsNone(patient)

    def test_patient_model_defaults(self):
        """Verify that the created_at timestamp is automatically generated."""
        save_patient("P105", "Timestamp Test", 50, "Other")
        patient = get_patient("P105")
        
        self.assertIsNotNone(patient.created_at)

if __name__ == '__main__':
    unittest.main()