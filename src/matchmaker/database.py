import json
import chromadb
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .config import LAWYERS_JSON, CHROMA_DIR, CACHE_DIR, EMBEDDING_MODEL

def _setup_embeddings():
    """Internal helper to initialize CPU embeddings"""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL, device="cpu", cache_folder=CACHE_DIR 
    )
    Settings.llm = None 

def update_lawyer_database():
    """Reads lawyers.json and pre-computes embeddings into ChromaDB."""
    _setup_embeddings()
    
    with open(LAWYERS_JSON, "r", encoding="utf-8") as f:
        lawyers_data = json.load(f)

    documents = [
        Document(
            text=lawyer["profile_text"], 
            metadata={k: v for k, v in lawyer.items() if k != "profile_text"}
        )
        for lawyer in lawyers_data
    ]

    db = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = db.get_or_create_collection("lawyer_profiles")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print("✅ Lawyer DB Updated!")

def get_ranked_lawyers(client_keywords, max_budget, top_k=5):
    """Searches the DB and applies the stakeholder ranking formula."""
    _setup_embeddings()
    
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = db.get_collection("lawyer_profiles")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = index.as_retriever(similarity_top_k=top_k)
    
    nodes = retriever.retrieve(client_keywords)
    ranked_candidates = []

    for node in nodes:
        meta = node.metadata
        hourly_rate = meta.get("hourly_rate", 9999)
        
        # 1. Hard Filter
        if hourly_rate > max_budget: continue 

        # 2. Formula
        relevance = node.score * 100 
        experience = meta.get("experience_years", 0) * 1.5
        availability = 15 if meta.get("available_immediately") else 0
        total_score = relevance + experience + availability

        ranked_candidates.append({
            "name": meta.get("name"),
            "specialties": meta.get("specialties"),
            "rate": hourly_rate,
            "total_score": round(total_score, 1)
        })

    ranked_candidates.sort(key=lambda x: x["total_score"], reverse=True)
    return ranked_candidates