import re

def classify_query(query: str) -> str:
    """
    Looks at the user's question and decides if it is looking for a 
    specific name (Entity) or a broader legal concept (Semantic).
    """
    entity_keywords = [r"who is", r"name the", r"which party", r"parties involved", r"najmul hasan"]
    
    query_lower = query.lower()
    for pattern in entity_keywords:
        if re.search(pattern, query_lower):
            return "entity"
            
    # If no specific names/keywords trigger, default to conceptual search
    return "semantic"