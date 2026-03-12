import json
# ragas_eval.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_integration.chains.rag_chains import DiagnosisRAG

from langchain_mistralai import MistralAIEmbeddings

api_key = os.getenv('MISTRAL_API_KEY')

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
)


# --- 1. YOUR TEST QUERIES ---
sample_queries = [
    "I have a persistent cough, shortness of breath, and wheezing.",
    "I have chest tightness, difficulty breathing, and I wake up at night coughing.",
    "I have a productive cough with green mucus, fever, and chest pain when breathing.",
    "I have been coughing for 3 weeks, lost weight, and have night sweats.",
    "I have sudden sharp chest pain, shortness of breath, and my lips look bluish.",
    "I have a runny nose, constant sneezing, sore throat and feel very tired.",
    "I have a cough, high fever, breathlessness and thick mucus when I cough.",
    "I have chills, sweating, fatigue, weight loss and blood in my sputum.",
    "I have chest pain, fast heart rate, and I am coughing up rusty coloured mucus.",
    "I have a persistent cough with thick mucus, wheezing, and frequent lung infections since childhood.",
    "I have a headache, runny nose, sinus pressure, loss of smell and muscle pain.",
    "I have fatigue, breathlessness, mucoid sputum and a family history of lung disease.",
    "I have high fever, chills, sweating and my phlegm has changed colour.",
    "I have a cough, fever, and I feel generally very unwell with chest discomfort.",
    "I have been sneezing continuously, my eyes are red and my throat is irritated."
]


expected_responses = [
    "Diagnosis: Bronchial Asthma. Recommended Test: Spirometry and Peak Flow Measurement.",
    "Diagnosis: Bronchial Asthma. Recommended Test: Spirometry and Peak Flow Measurement.",
    "Diagnosis: Pneumonia. Recommended Test: Chest X-Ray and Sputum Culture.",
    "Diagnosis: Tuberculosis. Recommended Test: Sputum Test, Chest X-Ray, and TB Skin Test.",
    "Diagnosis: Pneumonia. Recommended Test: Chest X-Ray and Sputum Culture.",
    "Diagnosis: Common Cold. Recommended Test: Physical Exam (No specific test required).",
    "Diagnosis: Bronchial Asthma. Recommended Test: Spirometry and Peak Flow Measurement.",
    "Diagnosis: Tuberculosis. Recommended Test: Sputum Test, Chest X-Ray, and TB Skin Test.",
    "Diagnosis: Pneumonia. Recommended Test: Chest X-Ray and Sputum Culture.",
    "Diagnosis: Cystic Fibrosis. Recommended Test: Sweat Chloride Test and Genetic Testing.",
    "Diagnosis: Common Cold. Recommended Test: Physical Exam (No specific test required).",
    "Diagnosis: Bronchial Asthma. Recommended Test: Spirometry and Peak Flow Measurement.",
    "Diagnosis: Tuberculosis. Recommended Test: Sputum Test, Chest X-Ray, and TB Skin Test.",
    "Diagnosis: Pneumonia. Recommended Test: Chest X-Ray and Sputum Culture.",
    "Diagnosis: Common Cold. Recommended Test: Physical Exam (No specific test required)."
]

# --- 2. RUN YOUR PIPELINE AND COLLECT OUTPUTS ---
rag = DiagnosisRAG()

responses = []
retrieved_contexts = []

for query in sample_queries:
    print(f"Running query: {query[:50]}...")
    result = rag.diagnose(query)
    
    responses.append(result["diagnosis"])
    # RAGAS expects a list of context strings per query
    raw_contexts = result["retrieved_context"].split("\n\n")
    clean_contexts = [ctx.split("] ", 1)[-1] if "] " in ctx else ctx for ctx in raw_contexts]
    retrieved_contexts.append(clean_contexts)

print("✅ All queries processed.")

# --- 3. BUILD RAGAS DATASET ---
data = {
    "question": sample_queries,
    "answer": responses,
    "contexts": retrieved_contexts,
    "ground_truth": expected_responses
}

with open("evaluation/eval_data.json", "w") as f:
    json.dump(data, f)

print("Data saved to eval_data.json")