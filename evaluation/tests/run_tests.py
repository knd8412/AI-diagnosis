import os
import sys
import time
from dotenv import load_dotenv

# --- ROBUST PATH LOGIC ---
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, base_dir)

try:
    from llm_integration.chains.rag_chains import DiagnosisRAG
except ImportError as e:
    print(f"❌ Error: Could not find 'llm_integration'.")
    sys.exit(1)

def run_checks():
    load_dotenv(os.path.join(base_dir, '.env'))
    
    results = []
    print("="*50)
    print("      🏥 RAG SYSTEM - SYSTEM HEALTH CHECK")
    print("="*50)

    # 1. CHECK API KEYS
    print("\n[1/4] Checking Environment Variables...")
    mistral_key = os.getenv("MISTRAL_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    
    if mistral_key and pinecone_key:
        results.append(("API Keys Configured", "✅ PASS"))
    else:
        results.append(("API Keys Configured", "❌ FAIL (Check .env)"))

    # 2. INITIALIZE RAG
    print("[2/4] Initializing RAG Pipeline...")
    try:
        start_time = time.time()
        rag = DiagnosisRAG()
        init_time = time.time() - start_time
        results.append(("Pipeline Initialization", f"✅ PASS ({init_time:.2f}s)"))
    except Exception as e:
        results.append(("Pipeline Initialization", f"❌ FAIL ({str(e)[:50]}...)"))
        rag = None

    if rag:
        # 3 & 4. TEST PIPELINE EXECUTION (RETIREVAL + LLM)
        print("[3/4] Testing End-to-End Pipeline...")
        try:
            # We run a single diagnosis to check both retrieval and response at once
            res = rag.diagnose("I have a persistent cough and fever.")
            
            # Check LLM Output
            if "diagnosis" in res and len(res["diagnosis"]) > 0:
                results.append(("LLM Response Generation", "✅ PASS"))
            else:
                results.append(("LLM Response Generation", "❌ FAIL (Empty response)"))

            # Check Pinecone Retrieval via the 'retrieved_context' key
            if "retrieved_context" in res and len(res["retrieved_context"]) > 0:
                results.append(("Pinecone Retrieval", "✅ PASS"))
            else:
                results.append(("Pinecone Retrieval", "❌ FAIL (No context found)"))
                
        except Exception as e:
            results.append(("End-to-End Pipeline", f"❌ FAIL ({str(e)[:50]}...)"))
    else:
        results.append(("End-to-End Pipeline", "⏭️ SKIPPED"))

    # --- SUMMARY TABLE ---
    print("\n" + "="*50)
    print(f"{'TEST NAME':<30} | {'RESULT':<15}")
    print("-" * 50)
    for name, res in results:
        print(f"{name:<30} | {res:<15}")
    print("="*50)

    if all("✅" in r[1] for r in results):
        print("\n🎉 ALL SYSTEMS GO! Your project is ready for submission.")
    else:
        print("\n⚠️ SOME CHECKS FAILED. Please review the errors above.")

if __name__ == "__main__":
    run_checks()