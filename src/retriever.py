import os
import chromadb
from typing import List, Optional
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Import your new control center!
import config

class LegalPositionPostprocessor(BaseNodePostprocessor):
    """Custom Reranker: Boosts the scores of the first and last pages of legal documents."""
    
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[dict] = None
    ) -> List[NodeWithScore]:
        for node in nodes:
            # Safely get the page number (default to middle if missing)
            page_num = node.metadata.get("page_number", 3) 
            
            # Boost Page 1 (Contains the actual Case Name and Parties)
            if page_num == 1:
                node.score += 0.5 
            # Boost Later Pages (Contains the Final Ruling/Order)
            elif page_num >= 4:
                node.score += 0.3
                
        # Re-sort the chunks so the boosted ones go to the top for the LLM to read first
        nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        return nodes

def setup_models():
    """Initializes the AI models so they can be used anywhere in the app."""
    config.logger.info("Booting up InLegalBert2 on CUDA...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="amixh/sentence-embedding-model-InLegalBERT-2",
        device="cuda",
        cache_folder=config.CACHE_DIR 
    )
    
    # --- NEW: The Google OpenAI-Compatibility Hack ---
    config.logger.info("Connecting to Google AI Studio (Gemma 31B) via OpenAI Adapter...")
    Settings.llm = OpenAILike(
        model="models/gemma-4-31b-it", 
        api_key=config.GEMINI_API_KEY,
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/", 
        temperature=0.0, 
        max_tokens=256,
        is_chat_model=True,       # Tells LlamaIndex how to format the prompt
        context_window=32768,     # Manually sets the context limit so it doesn't crash
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
    dense_retriever = index.as_retriever(similarity_top_k=15)
    nodes = list(storage_context.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=15)
    
    # 4. Fuse them together
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        similarity_top_k=10,
        num_queries=1,
        mode="reciprocal_rerank",
    )
    
    legal_qa_prompt = PromptTemplate(
        "[INST] TASK: Answer the query STRICTLY using the context.\n"
        "RULES:\n"
        "- NO PRECEDENTS: Ignore historical cases and quotes cited inside the text.\n"
        "- LOCATE RULING: The actual final decision and Case Crime numbers are at the very end or beginning.\n"
        "- FALLBACK: If answer is absent, reply EXACTLY: 'Insufficient evidence.'\n"
        "- CONCISENESS: Output maximum 3 sentences. No pleasantries or introductory filler.\n\n"
        "CONTEXT:\n{context_str}\n\n"
        "QUERY:\n{query_str}\n[/INST]"
    )
    # 5. Add the Reranker to grade the results
    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=12)
    positional_reranker = LegalPositionPostprocessor() # <-- NEW

    # 6. Package it all into the final engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker, positional_reranker], # <-- ADDED HERE 
        llm=Settings.llm
    )
    
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": legal_qa_prompt}
    )

    return hybrid_retriever, query_engine