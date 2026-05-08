import re

def classify_query(query: str) -> str:
    """
    Looks at the user's question and decides if it is looking for a 
    specific name (Entity) or a broader legal concept (Semantic).
    """
    entity_keywords = [

    # PERSON / PARTY IDENTIFICATION
    r"who is",
    r"name the",
    r"identify",
    r"which party",
    r"parties involved",
    r"petitioner",
    r"respondent",
    r"complainant",
    r"accused",
    r"appellant",
    r"opposite party",
    r"counsel",
    r"advocate",
    r"judge",
    r"bench",

    # CASE IDENTIFIERS
    r"case number",
    r"case crime",
    r"fir number",
    r"fir no",
    r"citation",
    r"neutral citation",
    r"crime no",
    r"petition number",

    # DATES / LOCATIONS
    r"date of",
    r"when was",
    r"where was",
    r"police station",
    r"district",
    r"court no",
    r"which court",

    # LEGAL SECTIONS / ACTS
    r"under section",
    r"which section",
    r"ipc",
    r"crpc",
    r"dowry prohibition act",
    r"charges",
    r"sections involved",

    # DOCUMENT STRUCTURE LOOKUPS
    r"what is the name",
    r"what was the fir",
    r"what are the charges",
    r"who filed",
    r"who lodged",
    r"who heard",
    r"who represented",

]
    
    query_lower = query.lower()
    for pattern in entity_keywords:
        if re.search(pattern, query_lower):
            return "entity"
            
    # If no specific names/keywords trigger, default to conceptual search
    return "semantic"