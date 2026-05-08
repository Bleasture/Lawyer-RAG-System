import os
import logging
import tiktoken
from statistics import mean
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Fetch the Gemini key securely
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not found in .env file!")

# Get the main folder path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define where everything lives
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_storage")
LOCAL_STORAGE_DIR = os.path.join(BASE_DIR, "local_storage")
CACHE_DIR = os.path.join(BASE_DIR, "my_local_cache")

# The path to your LLM
MISTRAL_PATH = r"C:\SeamRag\seamrag\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Setup standard logging so all files can use it
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Lawyer-RAG")

# ==================================================
# TOKEN DEBUGGING CONFIG
# ==================================================

encoding = tiktoken.get_encoding("cl100k_base")

session_prompt_tokens = []
session_response_tokens = []
session_total_tokens = []

def estimate_tokens(text: str) -> int:
    """
    Rough token estimation using OpenAI tokenizer encoding.
    Works well enough for local/debug usage.
    """
    if not text:
        return 0
    return len(encoding.encode(text))