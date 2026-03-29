from llm_integration.service.client import get_llm
from llm_integration.prompts.diagnosis_prompts import symptom_extraction_prompt
import json


class SymptomExtractor:
    def __init__(self):
        self.llm = get_llm()
    
    def extract(self, user_input: str):
        """
        Extract structured symptoms from natural language
        
        Args:
            user_input: Patient description
            
        Returns:
            dict: Structured symptom data
        """
        formatted_prompt = symptom_extraction_prompt.format(user_input=user_input)
        response = self.llm.invoke(formatted_prompt)
        
        try:
            # remove markdown code blocks if present
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw_response": response.content, "error": "Failed to parse JSON"}