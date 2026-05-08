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

            # ==================================================
            # TOKEN DEBUGGING
            # ==================================================

            retrieved_context = ""

            try:
                for node in response.source_nodes:
                    retrieved_context += node.text + "\n"
            except Exception:
                pass

            # Approximate full generation prompt
            constructed_prompt = f"""[INST] You are an expert Indian Legal AI Assistant. Your task is to answer the user's query based STRICTLY on the provided 'Retrieved Legal Context'.

            CRITICAL RULES:
            1. NO HALLUCINATION: Only use facts explicitly stated in the text.
            2. STRICT TERMINOLOGY: Do not confuse 'Sections' (statutory laws), 'Articles' (constitutional laws), and 'Case Laws' (precedents). If the user asks for 'Articles' and only 'Sections' are present, you must clarify this distinction.
            3. NEGATIVE CONSTRAINT: If the answer cannot be found in the context, reply EXACTLY with: "Insufficient evidence in the retrieved text."

            Retrieved Legal Context:
            {retrieved_context}

            User Query:
            {user_query}
            [/INST]"""

            prompt_tokens = config.estimate_tokens(constructed_prompt)
            response_tokens = config.estimate_tokens(response.response)
            total_tokens = prompt_tokens + response_tokens

            # Store session stats
            config.session_prompt_tokens.append(prompt_tokens)
            config.session_response_tokens.append(response_tokens)
            config.session_total_tokens.append(total_tokens)

            # ==================================================
            # MAIN OUTPUT
            # ==================================================

            print(f"\n AI Assistant: {response.response}")

            # ==================================================
            # TOKEN DEBUG OUTPUT
            # ==================================================

            print("\n================ TOKEN DEBUG ================")
            print(f"Prompt Tokens   : {prompt_tokens}")
            print(f"Response Tokens : {response_tokens}")
            print(f"Total Tokens    : {total_tokens}")

            print(
                f"Average Session Tokens : "
                f"{sum(config.session_total_tokens)/len(config.session_total_tokens):.2f}"
            )

            print("=============================================\n")

            config.logger.info(
                f"Query completed in {end_time - start_time:.2f} seconds\n"
            )
            
        except Exception as e:
            config.logger.error(f"Generation failed: {e}")

if __name__ == "__main__":
    start_chat()