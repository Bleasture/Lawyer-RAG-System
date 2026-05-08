import os
from dotenv import load_dotenv

# Resolve paths dynamically from the module's location
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, "../../"))

# --- NEW: Securely load the API Key ---
# Explicitly point to the .env file in the root folder
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Paths
LAWYERS_JSON = os.path.join(PROJECT_ROOT, "data", "lawyers.json")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_storage")
CACHE_DIR = os.path.join(PROJECT_ROOT, "my_local_cache")

# Models
#TRANSLATOR_MODEL = r"C:\Project\seamrag\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"