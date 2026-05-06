from .summarizer import summarize_case
from .engine import generate_strategy

# ==========================================
# 🚀 PRODUCTION ARCHITECTURE (FINAL MODULE)
# ==========================================
def process_two_cases(old_case_raw_text, current_case_raw_text):
    """
    FINAL MODULE LOGIC: 
    Takes two raw texts (e.g., extracted via PyMuPDF when the lawyer uploads them in the UI).
    No Vector DB is used here. It is a pure 1-to-1 comparison.
    """
    print("\n[PHASE 1] ⏳ Summarizing Old Precedent Case...")
    old_summary = summarize_case(old_case_raw_text, "Old Case")
    
    print("[PHASE 1] ⏳ Summarizing Current Active Case...")
    current_summary = summarize_case(current_case_raw_text, "Current Case")
    
    print("\n[PHASE 2] 🧠 Synthesizing Legal Strategy...")
    strategy = generate_strategy(old_summary, current_summary)
    
    print("\n" + "="*60)
    print("📜 FINAL STRATEGY REPORT")
    print("="*60)
    print(strategy)
    print("="*60 + "\n")
    
    return strategy


# ==========================================
# 🧪 TESTING HARNESS (Using Vector DB to save time)
# ==========================================
def run_test_with_db():
    """
    Pulls a chunk from your existing DB to act as the 'Old Case'
    so you don't have to type/upload it manually during backend testing.
    """
    import os
    import chromadb
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    print("[TEST] 🔍 Fetching a mock case from ChromaDB...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHROMA_DIR = os.path.join(BASE_DIR, "../../chroma_storage")
    CACHE_DIR = os.path.join(BASE_DIR, "../../my_local_cache")

    # Load embedder just for the test
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu", cache_folder=CACHE_DIR
    )

    db = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = db.get_collection("legal_cases")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Let's grab some text about 'termination' to act as our Old Case
    retriever = index.as_retriever(similarity_top_k=2)
    nodes = retriever.retrieve("contract termination notice")

    if not nodes:
        print("❌ DB is empty or no results found.")
        return

    # Combine the retrieved DB chunks to simulate an Old Case document
    mock_old_case_from_db = "\n...\n".join([n.text for n in nodes])
    file_name = nodes[0].metadata.get('file_name', 'Unknown DB File')
    print(f"[TEST] ✅ Grabbed text from '{file_name}' to use as Old Case.")

    # Hardcode a quick Current Case for the test
    mock_current_case = """
    INTAKE FILE: Our client wants to terminate their Joint Venture contract immediately. 
    They are unhappy with the partner. They have not given any written notice yet, they just 
    sent an angry email. The partner is threatening to sue for breach of contract.
    """

    # Unload embedder so Qwen gets all the VRAM
    Settings.embed_model = None 

    # PASS BOTH TO THE PRODUCTION FUNCTION!
    process_two_cases(mock_old_case_from_db, mock_current_case)


if __name__ == "__main__":
    # When you are ready to plug this into a UI, you will delete this __main__ block
    # and just import `process_two_cases` into your web server API.
    run_test_with_db()