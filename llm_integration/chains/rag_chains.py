from llm_integration.service.client import get_llm
from llm_integration.prompts.diagnosis_prompts import diagnosis_prompt
from llm_integration.retrieval.pinecone_client import get_pinecone_index
from llm_integration.retrieval.query_embedder import embed_query


class DiagnosisRAG:
    def __init__(self):
        """
        Initialize RAG chain with LLM and Pinecone connection
        """
        self.llm = get_llm()
        self.index, self.namespace = get_pinecone_index()

    def retrieve_context(self, query: str, top_k=5):
        """
        Retrieve relevant medical documents from Pinecone

        Args:
            query: Patient symptom description
            top_k: Number of documents to retrieve

        Returns:
            str: Combined context from retrieved documents
        """
        # Convert query to embedding
        query_embedding = embed_query(query)

        # Search Pinecone for similar documents
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True
        )

        # Extract text from results
        contexts = []
        for match in results['matches']:
            text = match['metadata'].get('page_content', '')
            score = match['score']
            contexts.append(f"[Relevance: {score:.2f}] {text}")

        return "\n\n".join(contexts)

    def diagnose(self, patient_query: str, chat_history=""):
        """
        Full RAG pipeline: retrieve relevant docs then generate diagnosis

        Args:
            patient_query: Patient's symptom description
            chat_history: Previous conversation context

        Returns:
            dict: Diagnosis and retrieved context
        """
        # Retrieve relevant medical information
        context = self.retrieve_context(patient_query)

        # Format prompt with retrieved context
        formatted_prompt = diagnosis_prompt.format(
            context=context,
            chat_history=chat_history,
            question=patient_query
        )

        # Generate diagnosis using LLM
        response = self.llm.invoke(formatted_prompt)

        return {
            "diagnosis": response.content,
            "retrieved_context": context
        }