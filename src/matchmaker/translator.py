from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from .config import GEMINI_API_KEY

def setup_translator():
    """Initializes the Gemma 4 31B model via the cloud API."""
    if not GEMINI_API_KEY:
        print("⚠️ ERROR: GEMINI_API_KEY not found in .env file!")

    Settings.llm = OpenAILike(
        model="models/gemma-4-31b-it", 
        api_key=GEMINI_API_KEY,
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/", 
        temperature=0.0,          # Keeps the extraction strict and deterministic
        max_tokens=50,            # Hard limit so it doesn't ramble
        is_chat_model=True,       
        context_window=32768,     
    )

def translate_client_complaint(client_text):
    prompt = f"""
You are an expert legal intake paralegal.

Extract ONLY:
- areas of law
- legal claims
- legal concepts

RULES:
- Output ONLY comma-separated legal terms.
- DO NOT explain reasoning.
- DO NOT reveal thoughts.
- DO NOT include analysis.
- NO markdown.
- NO extra text.

Client Complaint:
\"{client_text}\"

Legal Terms:
"""

    try:
        response = Settings.llm.complete(prompt).text.strip()
        if "</thought>" in response:
            response = response.split("</thought>")[-1].strip()
        return response
    except Exception as e:
        print(f"⚠️ Translation Error: {e}")
        return client_text # Fallback