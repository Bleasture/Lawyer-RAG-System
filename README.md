# Installation
### 1. Clone Repository
```bash
git clone https://github.com/Bleasture/Lawyer-RAG-System.git
cd Lawyer-RAG-System
```
### 2. Create Virtual Environment
```bash
python -m venv LawyerSystem
LawyerSystem\\Scripts\\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### Environment Variables
Create a .env file in the project root:
```bash
GEMINI_API_KEY=your_google_ai_studio_api_key
```
## Running the Legal RAG Summarizer

### Step 1 — Add Legal PDFs
Place legal judgments inside /data

### Step 2 — Ingest Documents
Run:
```bash
python -m src.ingest
```
### Step 3 — Run Legal Query Engine
Run:
```bash
python -m src.query
```
Example queries:
```bash
Summarize the whole case
Who is the petitioner?
Which IPC sections were invoked?
Why was the FIR quashed?
```

## Running the Lawyer Matchmaker

### Step 1 — Add Lawyers
Edit: data/lawyers.json
Example:
```json
[
  {
    "name": "Alice Chen",
    "specialties": "Real Estate Law, Tenant Rights",
    "experience_years": 10,
    "hourly_rate": 250,
    "available_immediately": true,
    "profile_text": "Experienced tenant rights and eviction lawyer."
  }
]
```
### Step 2 — Build Lawyer Vector Database
Run once whenever lawyers.json changes:
```bash
python -m src.matchmaker.update_db
```
### Step 3 — Run Matchmaker
```bash
python -m src.matchmaker.main
```
Example Complaint:
```bash
My landlord changed the locks and threw my belongings outside.
```
Example Output:
```bash
[1] Alice Chen
Specialties: Real Estate Law, Tenant Rights
```

# Models Used:

| Component             | Model                  |
| --------------------- | ---------------------- |
| Legal Embeddings      | InLegalBERT-2          |
| Reranker              | BAAI/bge-reranker-base |
| Matchmaker Embeddings | all-MiniLM-L6-v2       |
| Legal Generation      | Gemma 4 31B IT         |
| OCR                   | Tesseract OCR          |
