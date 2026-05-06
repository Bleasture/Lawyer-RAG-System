from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import Settings
from .config import STRATEGIST_MODEL

_LLM_INITIALIZED = False

def setup_llm():
    global _LLM_INITIALIZED
    
    if not _LLM_INITIALIZED:
        Settings.llm = LlamaCPP(
            model_path=STRATEGIST_MODEL,
            temperature=0.1,  
            max_new_tokens=1024, 
            context_window=4096,
            model_kwargs={"n_gpu_layers": 25},
            verbose=False
        )
        _LLM_INITIALIZED = True # Mark it as loaded!

def summarize_case(case_text, case_label="Case"):
    """Compresses a dense legal document into strict facts and rulings."""
    setup_llm()
    
    prompt = f"""You are a brilliant appellate paralegal. Read the following raw case file and extract the critical information.
Ignore all boilerplate text, formatting errors, and irrelevant procedural history.

{case_label} Text:
"{case_text}"

Output your summary strictly in this format:
FACTS: [Bullet points of what actually happened]
ISSUES: [The core legal questions the judge had to answer]
OUTCOME/RULING: [How the judge ruled and why (if applicable)]
"""
    try:
        return Settings.llm.complete(prompt).text.strip()
    except Exception as e:
        print(f"⚠️ Summarization Error for {case_label}: {e}")
        return "ERROR_EXTRACTING_SUMMARY"