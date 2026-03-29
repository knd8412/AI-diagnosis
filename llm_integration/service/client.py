import os
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI

load_dotenv()

def get_llm():
    """
    Initialise and return the LLM client configured for OpenRouter.

    Returns:
       Configured LLM instance
    """
    # Load API key from environment
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError("MISTRAL_API_KEY= not found in environment variables")

    # Initialize Mistral 
    llm = ChatMistralAI(
        model="mistral-large-latest",
        mistral_api_key=api_key,
        temperature=0.3,
        max_tokens=2000
    )

    return llm


if __name__ == "__main__":
    # Quick test
    print("LLM connection test\n")
    try:
        llm = get_llm()
        print("LLM initialised successfully\n")

        # Test that llm reponds
        print("Sending test message: 'Say hello!'\n")
        response = llm.invoke("Say hello!")

        print(f"LLM is connected\n")
        print(f"Response: {response.content}\n")

    except Exception as e:
        print(f"Error: {e}\n")
        import traceback

        traceback.print_exc()