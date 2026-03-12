from llm_integration.chains.rag_chains import DiagnosisRAG

print("Testing RAG Chain")

# NOTE: Before running this, make sure embedding service is running
# run from project root: python3 embedder/embedding_service.py

try:
    # initialize RAG chain
    rag = DiagnosisRAG()

    test_query = "Patient has persistent cough for 2 weeks with shortness of breath"

    print(f"Query: {test_query}\n")

    result = rag.diagnose(test_query)

    print("Diagnosis:")
    print(result["diagnosis"])

    print("\n" + "=" * 60)
    print("Retrieved context:")
    print(result["retrieved_context"][:500])

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()