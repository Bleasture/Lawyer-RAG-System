import os
import logging

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