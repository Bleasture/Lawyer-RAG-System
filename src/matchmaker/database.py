import json
import hashlib
import chromadb
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .config import LAWYERS_JSON, CHROMA_DIR, CACHE_DIR, EMBEDDING_MODEL

# ==================================================
# STABLE LAWYER ID GENERATOR
# ==================================================
def generate_lawyer_id(lawyer):
    """
    Creates a stable unique ID for each lawyer.
    Prevents duplicate indexing.
    """
    unique_string = (
        f"{lawyer.get('name', '')}_"
        f"{lawyer.get('specialties', '')}_"
        f"{lawyer.get('hourly_rate', '')}"
    )
    return hashlib.md5(unique_string.encode()).hexdigest()

def _setup_embeddings():
    """Internal helper to initialize CPU embeddings"""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL, device="cpu", cache_folder=CACHE_DIR 
    )
    #Settings.llm = None 

def update_lawyer_database():
    """
    Reads lawyers.json and safely updates ChromaDB
    without creating duplicate lawyer embeddings.
    """
    _setup_embeddings()
    with open(LAWYERS_JSON, "r", encoding="utf-8") as f:
        lawyers_data = json.load(f)

    db = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = db.get_or_create_collection(
        "lawyer_profiles",
        metadata={"hnsw:space": "cosine"}
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    existing_ids = set()
    # --------------------------------------------------
    # LOAD EXISTING IDS FROM CHROMA
    # --------------------------------------------------
    try:
        existing_data = collection.get()
        existing_ids = set(existing_data.get("ids", []))
    except Exception:
        pass

    new_documents = []

    for lawyer in lawyers_data:
        lawyer_id = generate_lawyer_id(lawyer)
        # ----------------------------------------------
        # SKIP DUPLICATES
        # ----------------------------------------------
        if lawyer_id in existing_ids:
            print(f"⚠️ Skipping duplicate lawyer: {lawyer.get('name')}")
            continue
        lawyer_metadata = {
            k: v for k, v in lawyer.items()
            if k != "profile_text"
        }
        lawyer_metadata["lawyer_id"] = lawyer_id
        doc = Document(
            text=lawyer["profile_text"],
            metadata=lawyer_metadata,
            id_=lawyer_id
        )
        new_documents.append(doc)

    if not new_documents:
        print("✅ No new lawyers to add. Database already up to date.")
        return
    VectorStoreIndex.from_documents(
        new_documents,
        storage_context=storage_context
    )
    
    print(f"✅ Added {len(new_documents)} new lawyers to DB!")

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

        if hourly_rate > max_budget: continue 

        # --- THE MATH FIX ---
        # node.score is Cosine Distance (0 is perfect, 1 is bad).
        # We subtract it from 1 to turn it into Cosine Similarity (1 is perfect, 0 is bad)
        relevance = (1 - node.score) * 100
        experience = meta.get("experience_years", 0) * 1.5
        availability = 15 if meta.get("available_immediately") else 0
        total_score = relevance + experience + availability

        specialties_text = str(meta.get("specialties", "")).lower()
        keywords_text = str(client_keywords).lower()

        specialization_boost = 0

        specialty_keywords = [
            "tenant",
            "real estate",
            "eviction",
            "family law",
            "criminal",
            "corporate",
            "divorce",
            "civil",
            "property",
            "housing"
        ]

        for keyword in specialty_keywords:
            if keyword in keywords_text and keyword in specialties_text:
                specialization_boost += 20
        
        total_score = (
            relevance +
            experience +
            availability +
            specialization_boost
        )

        ranked_candidates.append({
            "name": meta.get("name"),
            "specialties": meta.get("specialties"),
            "rate": hourly_rate,
            "relevance": round(relevance, 1),
            "experience": round(experience, 1),
            "availability": availability,
            "specialization_boost": specialization_boost,
            "total_score": round(total_score, 1)
        })

    ranked_candidates.sort(key=lambda x: x["total_score"], reverse=True)
    return ranked_candidates