import os
import sys
import json
import time
import logging
import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

# ---Set up logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "../chroma_storage")
CACHE_DIR = os.path.join(BASE_DIR, "../my_local_cache")
MISTRAL_PATH = r"C:\SeamRag\seamrag\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"

def setup_environment():
    logger.info("Booting up all-MiniLM-L6-v2 (Embeddings)...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda", cache_folder=CACHE_DIR 
    )
    
    logger.info("Booting up Mistral 7B (Generation)...")
    Settings.llm = LlamaCPP(
        model_path=MISTRAL_PATH, temperature=0.1, max_new_tokens=512, context_window=4096, 
        model_kwargs={"n_gpu_layers": -1}, verbose=False,
    )

def start_chat_interface():
    logger.info("Connecting to ChromaDB...")
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    vector_store = ChromaVectorStore(chroma_collection=db.get_collection("legal_cases"))
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # ---The Re-ranker  ---
    logger.info("Loading BGE Re-ranker...")
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base", top_n=3 # Cuts the top 10 down to the absolute best 3
    )
    
    print("\n LAWYER RAG SYSTEM ONLINE (Type 'quit' to exit)\n")

    while True:
        user_query = input(" Lawyer: ")
        if user_query.lower() in ['quit', 'exit', 'q']: break
        if not user_query.strip(): continue

        start_time = time.time()
        logger.info(f"Processing query: '{user_query}'")

        # --- MVP UPDATE: Removed hardcoded filter ---
        # Setting filters to None means the RAG will search across ALL ingested PDFs.
        # Later, your Firebase frontend can pass a specific filename here if needed.
        filters = None

        # Create the engine: Fetch 10 via vector search, then re-rank to 3
        query_engine = index.as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[reranker],
            filters=filters
        )
        
        try:
            response = query_engine.query(user_query)
            end_time = time.time()
            
            print(f"\n AI Assistant: {response.response}")
            logger.info(f"Query completed in {end_time - start_time:.2f} seconds")
            
            print("\n SOURCES CITED:")
            for i, node in enumerate(response.source_nodes):
                meta = node.metadata
                
                # Retrieve the plain strings saved by ingest.py
                parties = meta.get('parties', '').strip()
                clauses = meta.get('clauses', '').strip()
                obligations = meta.get('obligations', '').strip()
                
                print(f"  [{i+1}] Doc: {meta.get('file_name', 'Unknown')}")
                if parties: print(f"      Parties: {parties}")
                if clauses: print(f"      Clauses: {clauses}")
                if obligations: print(f"      Obligations: {obligations}")
            print("-" * 40 + "\n")
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")

if __name__ == "__main__":
    setup_environment()
    start_chat_interface()