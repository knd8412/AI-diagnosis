from langchain_core.prompts import PromptTemplate

#Prompt to extract symptoms from the text
symptom_extraction_prompt = PromptTemplate(
    input_variables=["user_input"], #Information that LLM receives
    template="""Extract structured medical information from the patient's description.

    Patient Input: {user_input}
    
    Extract and format as JSON:
    - symptoms: list of respiratory symptoms mentioned
    - patient_age: age if mentioned, otherwise null
    - patient_gender: gender if mentioned, otherwise null
    - duration: how long symptoms present, otherwise null
    - severity: mild/moderate/severe if mentioned, otherwise null
    
    Use medical terminology. Focus on the sympomts related to respiratory diseases, like: cough, shortness of breath, wheezing, chest pain, fever.
    
    JSON:"""
)

#Promt to give a diagnosis based on symptoms
diagnosis_prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are an AI assistant specializing in respiratory medicine diagnosis.

    Retrieved Medical Knowledge:
    {context}
    
    Previous Conversation:
    {chat_history}
    
    Current Patient Query: {question}
    
    Based ONLY on the retrieved medical knowledge above, provide:
    
    1. Ranked list of 3-5 possible respiratory conditions (most likely first)
    2. For each condition:
       - Match percentage with symptoms
       - Key clinical indicators from the retrieved sources
       - Brief explanation why it matches
    3. Recommended diagnostic tests (X-ray, spirometry, blood tests, etc.)
    
    Guidelines:
    - Use only information from the retrieved context
    - If retrieved information is insufficient, state what additional data is needed
    - Use professional medical language
    - Include confidence level: high/moderate/low
    
    Response:"""
)

#Prompt to explain why this diagnosis was given based on the symptoms and patient info
diagnosis_explanation_prompt = PromptTemplate(
    input_variables=["condition", "patient_symptoms", "context"],
    template="""Explain why a specific respiratory condition was suggested as a diagnosis.

    Condition in Question: {condition} 
    
    Patient Symptoms: {patient_symptoms}
    
    Medical Information from Database:
    {context}
    
    Provide a detailed explanation covering:
    
    1. Symptom Match 
       - Which patient symptoms align with this condition
       - Which symptoms are typical diagnostic indicators
    
    2. Clinical Reasoning
       - Why this condition is likely given the symptom pattern
       - What distinguishes it from similar conditions
    
    3. Confidence Assessment
       - How strongly symptoms match this condition
       - What additional information would increase/decrease confidence
    
    Use clear medical language.
    
    Explanation:"""
)