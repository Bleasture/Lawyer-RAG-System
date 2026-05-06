import os
import json
import io
import chromadb
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.llama_cpp import LlamaCPP

# --- 1. IMPORT OUR CONTROL CENTER ---
import config

# Windows requires pointing Python to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class SmartOCRReader:
    """Custom reader that extracts native text, but falls back to OCR for scanned pages and images."""
    def load_data(self, file, extra_info=None):
        doc = fitz.open(str(file))
        documents = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()

            if len(text) < 50:
                filename = os.path.basename(str(file))
                config.logger.info(f"      [OCR] Scanned content detected on {filename} (Page {page_num + 1}). Running Tesseract...")
                
                pix = page.get_pixmap(dpi=300) 
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)

            metadata = {
                "file_name": os.path.basename(str(file)), 
                "page_number": page_num + 1
            }
            if extra_info:
                metadata.update(extra_info)

            if text.strip():
                documents.append(Document(text=text, metadata=metadata))
                
        return documents

def setup_environment():
    """Initializes both the Embedding model and Mistral for ingestion."""
    config.logger.info("[1/5] Booting up InLegalBert2 on CUDA...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name = "amixh/sentence-embedding-model-InLegalBERT-2",
        device="cuda",
        cache_folder=config.CACHE_DIR 
    )
    
    config.logger.info("[2/5] Booting up Mistral 7B for Metadata Extraction...")
    Settings.llm = LlamaCPP(
        model_path=config.MISTRAL_PATH,
        temperature=0.0, 
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
    if not os.path.exists(config.DATA_DIR):
        config.logger.warning(f"Missing {config.DATA_DIR}")
        return

    config.logger.info(f"[3/5] Scanning for PDFs and Images in {config.DATA_DIR}...")
    
    reader = SimpleDirectoryReader(
        input_dir=config.DATA_DIR,
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
        config.logger.warning("No valid files found in the data directory.")
        return

    config.logger.info("[4/5] Document-Level Extraction and Chunking...")
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50, paragraph_separator="\n\n")
    all_nodes = []
    
    file_metadata_cache = {}

    for doc in documents:
        file_name = doc.metadata.get("file_name", "unknown_file")
        
        if file_name not in file_metadata_cache:
            intro_text = doc.text[:1500] 
            print(f"  -> Extracting global metadata for {file_name}...")
            
            doc_metadata = extract_metadata_with_llm(intro_text)
            file_metadata_cache[file_name] = {
                "parties": safe_list_to_string(doc_metadata.get("parties")),
                "clauses": safe_list_to_string(doc_metadata.get("clauses"))
            }
            
        global_parties = file_metadata_cache[file_name]["parties"]
        global_clauses = file_metadata_cache[file_name]["clauses"]
        
        doc_nodes = node_parser.get_nodes_from_documents([doc])
        
        for node in doc_nodes:
            node.metadata["parties"] = global_parties
            node.metadata["document_clauses"] = global_clauses
            all_nodes.append(node) 

    config.logger.info("[5/5] Generating embeddings and saving to ChromaDB & Docstore...")
    db = chromadb.PersistentClient(path=config.CHROMA_DIR)
    chroma_collection = db.get_or_create_collection("legal_cases")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    docstore = SimpleDocumentStore()
    docstore.add_documents(all_nodes) 
    
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore 
    )

    VectorStoreIndex(
        all_nodes, 
        storage_context=storage_context,
        show_progress=True
    )
    
    storage_context.persist(persist_dir=config.LOCAL_STORAGE_DIR)
    print("\n✅ Ingestion Complete! Your documents are now semantically enriched.")

if __name__ == "__main__":
    setup_environment()
    ingest_documents()