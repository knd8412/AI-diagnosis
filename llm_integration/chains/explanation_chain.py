from llm_integration.service.client import get_llm
from llm_integration.prompts.diagnosis_prompts import diagnosis_explanation_prompt


class DiagnosisExplainer:
    def __init__(self):
        self.llm = get_llm()
    
    def explain(self, condition: str, patient_symptoms: str, context: str):
        """
        Explain why a specific diagnosis was suggested
        
        Args:
            condition: Condition to explain
            patient_symptoms: Patient's symptoms
            context: Medical context
            
        Returns:
            str: Detailed explanation
        """
        formatted_prompt = diagnosis_explanation_prompt.format(
            condition=condition,
            patient_symptoms=patient_symptoms,
            context=context
        )
        response = self.llm.invoke(formatted_prompt)
        return response.content