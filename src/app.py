import time
import config
from router import classify_query
from retriever import setup_models, build_query_engine

def start_chat():
    # 1. Boot up models and get our engine
    setup_models()
    hybrid_retriever, query_engine = build_query_engine()
    
    print("\n==================================================")
    print(" ⚖️ LAWYER RAG SYSTEM ONLINE (Type 'quit' to exit)")
    print("==================================================\n")

    while True:
        user_query = input(" Lawyer: ")
        
        # Check if the user wants to exit
        if user_query.lower() in ['quit', 'exit', 'q']: 
            break
        if not user_query.strip(): 
            continue

        start_time = time.time()
        
        # 2. Route the query (The Traffic Cop)
        route = classify_query(user_query)
        config.logger.info(f"Query routed as: {route.upper()}")
        
        # 3. Adjust the weights on the fly based on the route
        if route == "entity":
            # 20% Semantic, 80% Keyword (Forces exact name matches to the top)
            hybrid_retriever.retriever_weights = [0.2, 0.8] 
        else:
            # 70% Semantic, 30% Keyword (Good default for legal reasoning)
            hybrid_retriever.retriever_weights = [0.7, 0.3] 

        # 4. Generate the Answer
        try:
            response = query_engine.query(user_query)
            end_time = time.time()
            
            print(f"\n AI Assistant: {response.response}")
            config.logger.info(f"Query completed in {end_time - start_time:.2f} seconds\n")
            
        except Exception as e:
            config.logger.error(f"Generation failed: {e}")

if __name__ == "__main__":
    start_chat()