from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

DB_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = f"sqlite:///{DB_DIR}/patients.db"

engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})

engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Patient(Base):
    __tablename__ = "patients"
    
    patient_id = Column(String, primary_key=True, index=True)
    name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Create the tables
Base.metadata.create_all(bind=engine)

# Helper for the UI/RAG logic
def get_patient(patient_id: str):
    db = SessionLocal()
    try:
        return db.query(Patient).filter(Patient.patient_id == patient_id).first()
    finally:
        db.close()

def save_patient(patient_id: str, name: str, age: int, gender: str):
    db = SessionLocal()
    try:
        db_patient = Patient(patient_id=patient_id, name=name, age=age, gender=gender)
        db.merge(db_patient) # Update if exists, insert if new
        db.commit()
    finally:
        db.close()

def get_all_patients():
    db = SessionLocal()
    try:
        return db.query(Patient).all()
    finally:
        db.close()