"""
Complete Example: Medical Diagnosis Assistant
This script demonstrates the full workflow from setup to diagnosis
"""

from medical_diagnosis_assistant import MedicalDiagnosisAssistant, print_diagnosis_results
import os
from dotenv import load_dotenv



def check_environment():
    """Check if environment is properly configured."""
    load_dotenv()
    
    pinecone_key = os.getenv('PINECONE_API_KEY')
    mistral_key = os.getenv('MISTRAL_API_KEY')
    
    print("Environment Check:")
    print("-" * 60)
    print(f"Pinecone API Key: {'✓ Found' if pinecone_key and pinecone_key != 'your_pinecone_api_key_here' else '✗ Not configured'}")
    print(f"Mistral API Key: {'✓ Found' if mistral_key and mistral_key != 'your_mistral_api_key_here' else '✗ Not configured'}")
    print("-" * 60)
    
    if not pinecone_key or pinecone_key == 'your_pinecone_api_key_here':
        print("\n⚠️  Warning: Pinecone API key not configured!")
        print("Please edit your .env file and add your Pinecone API key.")
        return False
    
    return True


def example_1_basic_diagnosis():
    """Example 1: Basic diagnosis with symptoms."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Diagnosis")
    print("="*70)
    
    # Initialize assistant
    assistant = MedicalDiagnosisAssistant()
    
    # Example symptoms
    symptoms = "I have a runny nose, sneezing, and a sore throat but no fever"
    
    print(f"\nPatient symptoms: {symptoms}")
    
    # Get diagnosis
    results = assistant.diagnose(symptoms, top_k=3)
    
    # Print results
    print_diagnosis_results(results)


def example_2_severe_symptoms():
    """Example 2: Diagnosis with severe symptoms."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Severe Symptoms Diagnosis")
    print("="*70)
    
    assistant = MedicalDiagnosisAssistant()
    
    symptoms = "I have severe chest pain, difficulty breathing, and my left arm hurts"
    
    print(f"\nPatient symptoms: {symptoms}")
    
    # Get diagnosis
    results = assistant.diagnose(symptoms, top_k=3)
    
    print_diagnosis_results(results)
    
    print("\n⚠️  ALERT: These symptoms may indicate a medical emergency!")
    print("         If you're experiencing these symptoms, call 911 immediately!")

def example_4_multiple_queries():
    """Example 4: Multiple diagnosis queries."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Multiple Diagnosis Queries")
    print("="*70)
    
    assistant = MedicalDiagnosisAssistant()
    
    # Different symptom scenarios
    scenarios = [
        "stomach pain, nausea, and diarrhea",
        "itchy eyes, sneezing, and clear runny nose",
        "severe headache with sensitivity to light",
        "painful urination and frequent need to pee"
    ]
    
    for i, symptoms in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i} ---")
        print(f"Symptoms: {symptoms}")
        
        results = assistant.diagnose(symptoms, top_k=2)
        
        # Print only top result for brevity
        if results:
            top_result = results[0]
            confidence_pct = top_result['confidence'] * 100
            print(f"\nTop diagnosis: {top_result['condition']} ({confidence_pct:.1f}% confidence)")
            print(f"Treatment: {top_result['treatment'][:100]}...")


def example_5_add_custom_condition():
    """Example 5: Add a custom medical condition."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Adding Custom Condition")
    print("="*70)
    
    assistant = MedicalDiagnosisAssistant()
    
    # Define a custom condition
    custom_condition = {
        "id": "custom_001",
        "condition": "Seasonal Affective Disorder (SAD)",
        "symptoms": "depression during winter months, low energy, oversleeping, weight gain, social withdrawal, loss of interest in activities",
        "treatment": "Light therapy (10,000 lux for 30 minutes daily), vitamin D supplements, regular exercise, counseling, antidepressants if needed",
        "additional_info": "Form of depression related to changes in seasons. Most common in fall/winter. Improves in spring/summer."
    }
    
    print("\nAdding custom condition to knowledge base...")
    print(f"Condition: {custom_condition['condition']}")
    
    assistant.add_new_condition(custom_condition)
    
    # Test search for this condition
    symptoms = "I feel very depressed and tired during winter, and I sleep too much"
    print(f"\nTesting with symptoms: {symptoms}")
    
    results = assistant.diagnose(symptoms, top_k=3)
    print_diagnosis_results(results)


def example_6_knowledge_base_stats():
    """Example 6: Get knowledge base statistics."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Knowledge Base Statistics")
    print("="*70)
    
    assistant = MedicalDiagnosisAssistant()
    
    stats = assistant.get_index_stats()
    
    print("\nKnowledge Base Information:")
    print(f"  Total medical conditions stored: {stats.get('total_vector_count', 0)}")
    print(f"  Embedding dimension: {stats.get('dimension', 0)}")
    print(f"  Index fullness: {stats.get('index_fullness', 0)}")
    
    # Show namespaces if any
    if 'namespaces' in stats:
        print(f"\n  Namespaces:")
        for ns, data in stats['namespaces'].items():
            print(f"    - {ns}: {data.get('vector_count', 0)} vectors")


def example_7_local_embeddings():
    print("\n" + "="*70)
    print("="*70)
    try:
        assistant = MedicalDiagnosisAssistant()
        
        symptoms = "headache and fever"
        print(f"\nPatient symptoms: {symptoms}")
        
        results = assistant.diagnose(symptoms, top_k=2)
        print_diagnosis_results(results)
        
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        print("Install with: pip install sentence-transformers")


def load_data_if_needed():
    """Load medical data if not already in Pinecone."""
    print("\n" + "="*70)
    print("LOADING MEDICAL KNOWLEDGE BASE")
    print("="*70)
    
    assistant = MedicalDiagnosisAssistant()
    
    stats = assistant.get_index_stats()
    vector_count = stats.get('total_vector_count', 0)
    
    if vector_count == 0:
        print("\nNo data found in knowledge base. Loading sample data...")
        
        # Check if medical_data.json exists
        if not os.path.exists('medical_data.json'):
            print("\nCreating sample medical data...")
            import create_medical_data
            create_medical_data.save_medical_data()
        
        # Load the data
        assistant.load_medical_knowledge('medical_data.json')
        print("\n✓ Medical knowledge base loaded!")
    else:
        print(f"\n✓ Knowledge base already contains {vector_count} medical conditions")
    
    return assistant


def run_all_examples():
    """Run all examples."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "MEDICAL DIAGNOSIS ASSISTANT" + " "*25 + "║")
    print("║" + " "*20 + "Complete Examples" + " "*29 + "║")
    print("╚" + "="*68 + "╝")
    
    # Check environment
    if not check_environment():
        print("\n❌ Please configure your environment first!")
        print("Run: python setup_project.py")
        return
    
    try:
        # Load data
        load_data_if_needed()
        
        # Run examples
        example_1_basic_diagnosis()
        input("\nPress Enter to continue to next example...")
        
        example_2_severe_symptoms()
        input("\nPress Enter to continue to next example...")
        
        
        example_4_multiple_queries()
        input("\nPress Enter to continue to next example...")
        
        example_5_add_custom_condition()
        input("\nPress Enter to continue to next example...")
        
        example_6_knowledge_base_stats()
        input("\nPress Enter to continue to next example...")        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your .env file is configured with valid API keys")
        print("2. Check your internet connection")
        print("3. Ensure Pinecone and Mistral services are accessible")
        print("4. Run: pip install -r requirements.txt")


def interactive_mode():
    """Interactive mode for custom symptom queries."""
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("\nEnter symptoms to get a diagnosis (or 'quit' to exit)")
    
    assistant = MedicalDiagnosisAssistant()
    
    while True:
        print("\n" + "-"*70)
        symptoms = input("Enter patient symptoms: ").strip()
        
        if symptoms.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not symptoms:
            print("Please enter some symptoms.")
            continue
        
        results = assistant.diagnose(symptoms, top_k=3)
        print_diagnosis_results(results)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        # Run interactive mode
        load_data_if_needed()
        interactive_mode()
    else:
        # Run all examples
        run_all_examples()
        
        # Offer interactive mode
        print("\nWould you like to try interactive mode? (y/n)")
        response = input("> ").strip().lower()
        if response in ['y', 'yes']:
            interactive_mode()