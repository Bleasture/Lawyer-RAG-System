from .translator import setup_translator, translate_client_complaint
from .database import update_lawyer_database, get_ranked_lawyers

def process_new_client(complaint_text, budget):
    print(f"\n🗣️ CLIENT: '{complaint_text}' | BUDGET: ${budget}/hr")
    
    print("⏳ Translating to legalese...")
    keywords = translate_client_complaint(complaint_text)
    print(f"⚖️ VECTOR: {keywords}\n")
    
    print("🔍 Searching and ranking lawyer database...")
    results = get_ranked_lawyers(keywords, budget)
    
    print("🏆 TOP MATCHES:")
    print("-" * 40)
    if not results:
        print("❌ No lawyers matched the criteria.")
    else:
        for i, lawyer in enumerate(results, 1):
            print(f"[{i}] {lawyer['name']} | Score: {lawyer['total_score']} pts | Rate: ${lawyer['rate']}/hr")
            print(f"    Specialties: {lawyer['specialties']}\n")

if __name__ == "__main__":
    # Optional: Run this once if you updated lawyers.json
    # update_lawyer_database() 
    
    setup_translator()
    
    # Simulate a web request hitting your API
    complaint = "My landlord changed the locks on my apartment while I was at work and threw all my clothes onto the street! He said I missed rent by 2 days, but this is insane!"
    budget = 300
    
    process_new_client(complaint, budget)