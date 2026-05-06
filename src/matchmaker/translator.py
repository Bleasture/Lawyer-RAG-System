from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import Settings
from .config import TRANSLATOR_MODEL

def setup_translator():
    Settings.llm = LlamaCPP(
        model_path=TRANSLATOR_MODEL,
        temperature=0.1, 
        max_new_tokens=100, 
        context_window=4096,
        model_kwargs={"n_gpu_layers": 25},
        verbose=False
    )

def translate_client_complaint(client_text):
    prompt = f"""You are an expert legal intake paralegal. 
Read the client's informal complaint and extract ONLY the formal legal areas of law, case types, and key legal concepts it represents.
Output ONLY a comma-separated list of legal terms. No conversational text.

Client Complaint: "{client_text}"

Formal Legal Terms:"""

    try:
        return Settings.llm.complete(prompt).text.strip()
    except Exception as e:
        print(f"⚠️ Translation Error: {e}")
        return client_text # Fallback