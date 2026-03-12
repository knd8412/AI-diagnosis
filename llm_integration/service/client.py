import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm():
    """
    Initialise and return the LLM client configured for OpenRouter.

    Returns:
        ChatOpenAI: Configured LLM instance
    """
    # Load API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    # Initialize ChatOpenAI with OpenRouter configuration
    llm = ChatOpenAI(
        model_name="arcee-ai/trinity-large-preview:free",  # Free tier for MVP
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3,  # Lower temp for more consistent medical responses
        max_tokens=500  # Adjust based on your needs
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