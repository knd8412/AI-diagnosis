from llm_integration.chains.symptom_extraction_chain import SymptomExtractor
from llm_integration.chains.rag_chains import DiagnosisRAG
from llm_integration.chains.explanation_chain import DiagnosisExplainer


class DiagnosisOrchestrator:
    """
    Smart orchestrator that routes queries to appropriate chains
    """
    def __init__(self):
        self.extractor = SymptomExtractor()
        self.rag = DiagnosisRAG()
        self.explainer = DiagnosisExplainer()
        self.last_diagnosis = None  # store last diagnosis for follow-ups
    
    def process(self, user_input: str, chat_history=""):
        """
        Smart routing: decides which chain to use based on query type
        
        Args:
            user_input: User query
            chat_history: Previous conversation
            
        Returns:
            dict: Response with type and content
        """
        # Check if it's a "why" question
        if self._is_explanation_query(user_input):
            return self._handle_explanation(user_input)
        
        # Otherwise it's a diagnosis request
        return self._handle_diagnosis(user_input, chat_history)
    
    def _is_explanation_query(self, query: str) -> bool:
        """
        Detect if query is asking for explanation
        """
        query_lower = query.lower()
        explanation_keywords = [
            "why", "explain", "how come", "reason", 
            "what makes you think", "why did you"
        ]
        return any(keyword in query_lower for keyword in explanation_keywords)
    
    def _handle_diagnosis(self, user_input: str, chat_history: str):
        """
        Full diagnosis workflow
        """
        # Extract symptoms
        symptoms = self.extractor.extract(user_input)
        
        # Run RAG diagnosis
        diagnosis_result = self.rag.diagnose(user_input, chat_history)
        
        # Store for potential follow-up questions
        self.last_diagnosis = {
            "symptoms": symptoms,
            "diagnosis": diagnosis_result["diagnosis"],
            "context": diagnosis_result["retrieved_context"]
        }
        
        return {
            "type": "diagnosis",
            "extracted_symptoms": symptoms,
            "diagnosis": diagnosis_result["diagnosis"],
            "retrieved_context": diagnosis_result["retrieved_context"]
        }
    
    def _handle_explanation(self, user_input: str):
        """
        Handle "why X?" questions
        """
        if not self.last_diagnosis:
            return {
                "type": "error",
                "message": "No previous diagnosis to explain. Please describe your symptoms first."
            }
        
        # Extract condition from query or use from last diagnosis
        condition = self._extract_condition(user_input)
        
        explanation = self.explainer.explain(
            condition=condition,
            patient_symptoms=str(self.last_diagnosis["symptoms"]),
            context=self.last_diagnosis["context"]
        )
        
        return {
            "type": "explanation",
            "condition": condition,
            "explanation": explanation
        }
    
    def _extract_condition(self, query: str) -> str:
        """
        Extract condition name from query like "why asthma?"
        """
        # Simple extraction - can be improved
        query_lower = query.lower()
        
        # Try to find condition name after "why"
        if "why" in query_lower:
            parts = query_lower.split("why")
            if len(parts) > 1:
                condition = parts[1].strip().replace("?", "").replace("did you suggest", "").strip()
                return condition.title()
        
        # Fallback: return first condition from last diagnosis
        if self.last_diagnosis:
            diagnosis_text = self.last_diagnosis["diagnosis"]
            # Extract first mentioned condition (simple approach)
            return "the suggested condition"
        
        return "unknown condition"