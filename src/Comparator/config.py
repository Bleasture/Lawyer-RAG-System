import os

# Resolve paths dynamically from the module's location
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, "../../"))

# Models (Using your strongest reasoning model for the Strategist)
STRATEGIST_MODEL = r"C:\Project\seamrag\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf"

GEMINI_API_KEY = "AIzaSyDhdoO3dpWySTHmv1AdP2IvS_inSB2fZPE"