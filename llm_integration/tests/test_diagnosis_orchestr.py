from llm_integration.chains.diagnosis_orchestration import DiagnosisOrchestrator

print("Testing Smart Orchestrator")
print("="*60)

orchestrator = DiagnosisOrchestrator()

# Test 1: Initial diagnosis
print("\nTest 1: Initial Diagnosis Request")
query1 = "I have a persistent cough and shortness of breath for 2 weeks"
result1 = orchestrator.process(query1)

print(f"Query: {query1}")
print(f"Type: {result1['type']}")
print(f"Symptoms: {result1.get('extracted_symptoms')}")
print(f"Diagnosis: {result1['diagnosis'][:200]}...")

# Test 2: Follow-up explanation
print("\n" + "="*60)
print("\nTest 2: Follow-up Why Question")
query2 = "Why did you suggest asthma?"
result2 = orchestrator.process(query2)

print(f"Query: {query2}")
print(f"Type: {result2['type']}")
print(f"Explanation: {result2.get('explanation', 'N/A')[:200]}...")