from llama_index.core import Settings
from .summarizer import setup_llm

def generate_strategy(old_case_summary, current_case_summary):
    """Compares the Old Case to the Current Case to draft a legal strategy."""
    setup_llm()
    
    # 1. THE PROMPT FIX: Added a strict stop marker to the template
    prompt = f"""You are a ruthless, highly strategic Defense Attorney. 
    You have two files on your desk. 

    1. OLD PRECEDENT CASE (How a judge previously ruled):
    {old_case_summary}

    2. OUR CURRENT CASE (The active dispute we are fighting):
    {current_case_summary}

    TASK:
    Based strictly on the Outcome/Ruling of the Old Case, draft a highly actionable, 3-point legal strategy to defend OUR CLIENT in Our Current Case. 

    CRITICAL RULES:
    1. KNOW YOUR ROLE: You represent "Our Client" in the Current Case. The other party is the "Opponent".
    2. NEVER PROSECUTE YOUR OWN CLIENT: Even if our client made a terrible mistake (like breaching a contract or failing to give notice), DO NOT advise attacking our client. 
    3. DAMAGE CONTROL: If our client is legally in the wrong based on the Old Case, your strategy must focus on damage control, finding loopholes, negotiating settlements, or distinguishing the cases to save our client from liability.

    Output format:
    ### ⚖️ Precedent Analysis
    [1 paragraph explaining how the two cases overlap, and honestly assessing if the precedent is helpful or harmful to our client]

    ### 🎯 3-Point Legal Strategy
    1. [Defense Strategy Point 1]
    2. [Defense Strategy Point 2]
    3. [Defense Strategy Point 3]

    [END OF STRATEGY]
    """
    try:
        response = Settings.llm.complete(prompt).text.strip()
        
        # 2. THE CODE FIX (The Guillotine)
        # If the LLM printed our end marker but kept talking, chop off the rest
        if "[END OF STRATEGY]" in response:
            response = response.split("[END OF STRATEGY]")[0]
            
        # 3. THE FAIL-SAFE
        # If it forgot the marker and just looped the header, keep only the first loop
        if response.count("### ⚖️ Precedent Analysis") > 1:
            parts = response.split("### ⚖️ Precedent Analysis")
            # parts[0] is empty space, parts[1] is the first good loop
            response = "### ⚖️ Precedent Analysis\n" + parts[1]
            
        return response.strip()
        
    except Exception as e:
        print(f"⚠️ Strategy Generation Error: {e}")
        return "ERROR_GENERATING_STRATEGY"