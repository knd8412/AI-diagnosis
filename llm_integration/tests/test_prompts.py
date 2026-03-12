from llm_integration.service.client import get_llm
from llm_integration.prompts.diagnosis_prompts import (
    symptom_extraction_prompt,
    diagnosis_prompt,
    diagnosis_explanation_prompt
)

print("Testing LLM Responses with Prompts")

# initialize LLM
llm = get_llm()

# first prompt test: symptom extraction
print("\nSymptom Extraction")

# mock input since not yet connected to vector store
test_input = "I'm a 45 year old male. I've been coughing for about a week now with some chest tightness and shortness of breath"

formatted_prompt = symptom_extraction_prompt.format(user_input=test_input)
print(f"Input: {test_input}\n")

response = llm.invoke(formatted_prompt)
print(f"\nResponse:\n{response.content}")

# second prompt test: diagnosis
print("\nDiagnosis Generation")

test_context = """
Asthma: A chronic respiratory condition characterized by airway inflammation and bronchospasm. 
Common symptoms include wheezing, shortness of breath, chest tightness, and coughing, especially at night.

Chronic Bronchitis: Inflammation of bronchial tubes causing persistent cough with mucus production.
Symptoms include cough lasting 3+ months, shortness of breath, chest discomfort.
"""

formatted_prompt = diagnosis_prompt.format(
    context=test_context,
    chat_history="",
    question="Patient has persistent cough for 1 week with chest tightness and shortness of breath"
)

response = llm.invoke(formatted_prompt)
print(f"\nResponse:\n{response.content}")

# third prompt test: explanation
print("\nDiagnosis Explanation")

formatted_prompt = diagnosis_explanation_prompt.format(
    condition="Asthma",
    patient_symptoms="cough for 1 week, chest tightness, shortness of breath",
    context=test_context
)

response = llm.invoke(formatted_prompt)
print(f"\nResponse:\n{response.content}")