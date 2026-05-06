import os
import chromadb
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Import your new control center!
import config

def setup_models():
    """Initializes the AI models so they can be used anywhere in the app."""
    config.logger.info("Booting up InLegalBert2 on CUDA...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="amixh/sentence-embedding-model-InLegalBERT-2",
        device="cuda",
        cache_folder=config.CACHE_DIR 
    )
    
    config.logger.info("Booting up Mistral 7B for Generation...")
    Settings.llm = LlamaCPP(
        model_path=config.MISTRAL_PATH,
        temperature=0.0, 
        max_new_tokens=512,
        context_window=4096,
        model_kwargs={"n_gpu_layers": -1}, 
        verbose=False,
    )

def build_query_engine():
    """Connects to the databases and sets up the Hybrid Search."""
    config.logger.info("Connecting to Vector and Keyword Storage...")
    
    # 1. Load Chroma (Semantic Search)
    db = chromadb.PersistentClient(path=config.CHROMA_DIR)
    vector_store = ChromaVectorStore(chroma_collection=db.get_collection("legal_cases"))
    
    # 2. Load Local Storage (BM25 Keyword Search)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=config.LOCAL_STORAGE_DIR
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, 
        storage_context=storage_context
    )
    
    # 3. Setup Both Retrievers
    dense_retriever = index.as_retriever(similarity_top_k=8)
    nodes = list(storage_context.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=8)
    
    # 4. Fuse them together
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        similarity_top_k=10,
        num_queries=1,
        mode="reciprocal_rerank",
    )
    
    # 5. Add the Reranker to grade the results
    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=3)

    # 6. Package it all into the final engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker], 
        llm=Settings.llm
    )
    
    return hybrid_retriever, query_engine