import os
import json
import io
import chromadb
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.llama_cpp import LlamaCPP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
CHROMA_DIR = os.path.join(BASE_DIR, "../chroma_storage")
CACHE_DIR = os.path.join(BASE_DIR, "../my_local_cache")

# UPDATE THESE PATHS
MISTRAL_PATH = r"C:\SeamRag\seamrag\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Windows requires pointing Python to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class SmartOCRReader:
    """Custom reader that extracts native text, but falls back to OCR for scanned pages and images."""
    def load_data(self, file, extra_info=None):
        # fitz handles both PDFs and raw image files (.jpg, .png) seamlessly
        doc = fitz.open(str(file))
        documents = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()

            # If the page has almost no text, assume it's a scanned image
            if len(text) < 50:
                filename = os.path.basename(str(file))
                print(f"      [OCR] Scanned content detected on {filename} (Page {page_num + 1}). Running Tesseract...")
                
                # Render the page as a high-res image (300 DPI) for accurate OCR
                pix = page.get_pixmap(dpi=300) 
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                
                # Run Tesseract
                text = pytesseract.image_to_string(img)

            metadata = {
                "file_name": os.path.basename(str(file)), 
                "page_number": page_num + 1
            }
            if extra_info:
                metadata.update(extra_info)

            # Ensure we only append pages that actually yielded text
            if text.strip():
                documents.append(Document(text=text, metadata=metadata))
                
        return documents


def setup_environment():
    """Initializes both the Embedding model and Mistral."""
    print("[1/5] Booting up all-MiniLM-L6-v2 on CUDA...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda",
        cache_folder=CACHE_DIR 
    )
    
    print("[2/5] Booting up Mistral 7B for Metadata Extraction...")
    Settings.llm = LlamaCPP(
        model_path=MISTRAL_PATH,
        temperature=0.1, 
        max_new_tokens=256,
        context_window=4096,
        model_kwargs={"n_gpu_layers": -1}, 
        verbose=False,
    )

def extract_metadata_with_llm(text_chunk):
    """Prompts Mistral to extract structured JSON from a text chunk."""
    prompt = f"""[INST] You are an expert legal data extractor. Read the text and extract the parties, clauses, and obligations into a strict JSON object. 
If a category is missing, use an empty list []. Do not include any text outside of the JSON block.

Format required:
{{
    "parties": ["party 1", "party 2"],
    "clauses": ["clause name"],
    "obligations": ["obligation 1"]
}}

Text to analyze:
{text_chunk}
[/INST]"""

    try:
        response = Settings.llm.complete(prompt).text
        clean_json = response.replace("```json", "").replace("```", "").strip()
        metadata = json.loads(clean_json)
        return metadata
    except json.JSONDecodeError:
        return {"parties": [], "clauses": [], "obligations": ["Extraction Error"]}
    except Exception as e:
        return {"parties": [], "clauses": [], "obligations": [f"Error: {str(e)}"]}

def safe_list_to_string(extracted_item):
    """Forces whatever the LLM hallucinates into a clean, comma-separated string."""
    if not extracted_item:
        return ""
    if isinstance(extracted_item, str):
        return extracted_item
    if isinstance(extracted_item, list):
        return ", ".join(str(i) for i in extracted_item)
    return str(extracted_item)

def ingest_documents():
    if not os.path.exists(DATA_DIR):
        print(f"⚠️ Missing {DATA_DIR}")
        return

    print(f"[3/5] Scanning for PDFs and Images in {DATA_DIR}...")
    
    # Register our SmartOCRReader to handle PDFs and common image formats
    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        required_exts=[".pdf", ".PDF", ".jpg", ".jpeg", ".png"],
        recursive=True,
        file_extractor={
            ".pdf": SmartOCRReader(),
            ".PDF": SmartOCRReader(),
            ".jpg": SmartOCRReader(),
            ".jpeg": SmartOCRReader(),
            ".png": SmartOCRReader(),
        }
    )
    documents = reader.load_data()
    
    if not documents:
        print("⚠️ No valid files found in the data directory.")
        return

    print("[4/5] Chunking text and Extracting Legal Metadata (This will take time)...")
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50, paragraph_separator="\n\n")
    nodes = node_parser.get_nodes_from_documents(documents)
    
    for i, node in enumerate(nodes):
        print(f"  -> Processing chunk {i+1}/{len(nodes)}...")
        extracted_data = extract_metadata_with_llm(node.text)
        
        node.metadata["parties"] = safe_list_to_string(extracted_data.get("parties"))
        node.metadata["clauses"] = safe_list_to_string(extracted_data.get("clauses"))
        node.metadata["obligations"] = safe_list_to_string(extracted_data.get("obligations"))

    print("[5/5] Generating embeddings and saving to ChromaDB...")
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = db.get_or_create_collection("legal_cases")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True
    )
    print("\n✅ Phase 4A Complete! Your documents are now semantically enriched.")

if __name__ == "__main__":
    setup_environment()
    ingest_documents()