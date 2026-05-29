AI-Powered Banking Support &amp; Fraud Intelligence System using NLP + RAG
# AI-Powered Banking Support & Fraud Intelligence System
│
├── data/
│   ├── 01_support_tickets.csv
│   ├── 02_transactions.csv
│   └── 04_qa_pairs.json
│
├── models/
│   ├── intent_vectorizer.pkl
│   ├── intent_model.pkl
│   ├── fraud_scaler.pkl
│   ├── fraud_le.pkl
│   └── fraud_model.pkl
│
├── training/
│   └── train_all.py
│
└── app.py   # Streamlit dashboard


##  Project Objectives
- Build an AI-enabled support system for banking queries.
- Detect customer **intent** (Fraud, Loan, KYC, General).
- Assess **fraud risk level** (High / Medium / Low) using transaction data.
- Detect **sentiment** (Positive / Neutral / Negative) to prioritize urgent queries.
- Use **RAG (Retrieval-Augmented Generation)** to provide policy-aligned answers by combining:
  - Predefined QA pairs
  - Official policy documents
  - Real-world resolution texts from support tickets

##  Setup Instructions

### 1. Environment
- Python 3.10+
- Recommended: Create a virtual environment
  python -m venv venv
  source venv/bin/activate   # Linux/Mac
  venv\Scripts\activate      # Windows

User Query
    |
    v
SentenceTransformer (Embedder) ---> FAISS Vector Database
                                      |
                                      v
                           Retrieve Top-k Relevant Chunks
                                      |
                                      v
QA Model (DistilBERT Question-Answering)
                                      |
                                      v
                                Final Answer
                                
Knowledge Base:
- QA Pairs
- Policy Documents
- Resolution Texts
