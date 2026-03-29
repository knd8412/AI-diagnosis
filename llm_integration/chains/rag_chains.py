from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from llm_integration.service.client import get_llm
from llm_integration.prompts.diagnosis_prompts import diagnosis_prompt, symptom_extraction_prompt
from llm_integration.retrieval.pinecone_client import get_pinecone_index
from llm_integration.retrieval.query_embedder import embed_query
from llm_integration.retrieval.chromaClient import get_patient_memory
from SQLdb.models import SessionLocal, Patient

class DiagnosisRAG:
    def __init__(self):
        """Initialize the RAG orchestration with LLM, Vector Stores, and Chains."""
        self.llm = get_llm()
        self.index, self.namespace = get_pinecone_index()
        # Initialize the PatientMemory wrapper class
        self.patient_memory = get_patient_memory()

        # Initialize LCEL Chains to handle variables correctly
        self.extraction_chain = symptom_extraction_prompt | self.llm | JsonOutputParser()
        self.diagnosis_chain = diagnosis_prompt | self.llm | StrOutputParser()

    def _get_patient_details(self, patient_id: str):
        """Fetch demographics from SQL and semantic history from ChromaDB."""
        # 1. SQL Data Fetching
        db = SessionLocal()
        p = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        db.close()
        
        sql_info = f"Patient: {patient_id}"
        if p:
            # Aligned with the 'age' and 'gender' fields in SQL 
            sql_info += f" (Gender: {p.gender}, Age: {p.age})"
        
        # ChromaDB Semantic Search
        history_str = self.patient_memory.search_history(
            patient_id=patient_id,
            query="prior respiratory symptoms and history",
            n_results=2
        )
        
        if not history_str:
            history_str = "No prior semantic history found."
        
        return f"{sql_info}\nPast Conversations: {history_str}"

    def retrieve_context(self, query: str, top_k=5):
        """Retrieve relevant medical documents from Pinecone."""
        query_embedding = embed_query(query)

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True
        )

        contexts = [match['metadata'].get('page_content', '') for match in results['matches']]
        return "\n\n".join(contexts)

    def diagnose(self, patient_id: str, patient_query: str, chat_history: str = ""):
        """

            Main orchestration: Extract Symptoms -> Retrieve Context -> Generate Diagnosis.

            Includes keyword argument 'chat_history' to match UI calls.

        """
        extracted_data = self.extraction_chain.invoke({"user_input": patient_query})
        symptoms_list = extracted_data.get("symptoms", [])

        # NEW: also pull pathology names directly from system notes as fallback
        xray_conditions = []
        if "[System Note:" in patient_query:
            import re
            matches = re.findall(r"findings of: ([^\]]+)\]", patient_query)
            for match in matches:
                # Extract just condition names, strip confidence scores
                conditions = [c.split("(")[0].strip() for c in match.split(",")]
                xray_conditions.extend(conditions)

        # Combine both sources for retrieval
        all_terms = symptoms_list + xray_conditions

        if not all_terms:
            medical_context = "No specific symptoms provided for medical retrieval."
        else:
            search_query = ", ".join(all_terms)
            medical_context = self.retrieve_context(search_query)

        patient_specific_context = self._get_patient_details(patient_id)

        # Generate diagnosis using the retrieved context and patient history
        response_text = self.diagnosis_chain.invoke({
            "context": medical_context,
            "chat_history": f"{patient_specific_context}\nRecent chat: {chat_history}",
            "question": patient_query
        })

        # Save to ChromaDB for future recall
        self.patient_memory.add_interaction(patient_id, "user", patient_query)
        self.patient_memory.add_interaction(patient_id, "assistant", response_text)

        return {
            "diagnosis": response_text,
            "retrieved_context": medical_context
        }