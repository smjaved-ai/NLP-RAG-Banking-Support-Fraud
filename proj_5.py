import streamlit as st
import pandas as pd
import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression


# -------------------------------
# Load models
# -------------------------------
import joblib, json, faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Intent classifier
intent_vectorizer = joblib.load("intent_vectorizer.pkl")
intent_model = joblib.load("intent_model.pkl")

# Fraud classifier
fraud_scaler = joblib.load("fraud_scaler.pkl")
fraud_le = joblib.load("fraud_le.pkl")
fraud_model = joblib.load("fraud_model.pkl")

# Sentiment detector
sentiment_model = pipeline("sentiment-analysis",
                           model="distilbert-base-uncased-finetuned-sst-2-english")

# -------------------------------
# RAG pipeline with QA + Policies
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load QA pairs
with open("04_qa_pairs.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)
questions = [item["question"] for item in qa_data]
answers = [item["answer"] for item in qa_data]

# Load policy documents
policy_files = [
    "fraud_handling_policy.txt",
    "kyc_policy.txt",
    "loan_processing_policy.txt",
    "refund_dispute_policy.txt"
]

policy_chunks = []
for file in policy_files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            # Split into smaller chunks for embedding
            chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
            policy_chunks.extend(chunks)
    except FileNotFoundError:
        print(f"⚠️ Policy file not found: {file}")

# Combine QA answers + policy chunks + Resolution text
df_tickets = pd.read_csv("01_support_tickets.csv")
resolution_texts = df_tickets["resolution_text"].dropna().tolist()

corpus = answers + policy_chunks + resolution_texts
corpus_embeddings = embedder.encode(corpus)

# Build FAISS index
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

# QA model
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def retrieve_answer(query, top_k=3):
    query_vec = embedder.encode([query])
    distances, indices = index.search(query_vec, top_k)
    return [corpus[i] for i in indices[0]]

def generate_response(query):
    retrieved_contexts = retrieve_answer(query, top_k=3)
    context = " ".join(retrieved_contexts)
    result = qa_model(question=query, context=context)
    return result["answer"]

# -------------------------------
# Fraud Risk Mapping
# -------------------------------
def map_risk(prob):
    if prob > 0.7:
        return "High"
    elif prob > 0.4:
        return "Medium"
    else:
        return "Low"

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("AI-Powered Banking Support & Fraud Intelligence System")

query = st.text_input("Enter customer query:")

if query:
    # Intent classification
    query_vec = intent_vectorizer.transform([query])
    intent = intent_model.predict(query_vec)[0]

    # Sentiment detection
    sentiment_result = sentiment_model(query)[0]
    sentiment = sentiment_result["label"]

    # RAG response
    rag_answer = generate_response(query)

        # Display results
    #st.subheader("Response")
    st.subheader("Intent & Sentiment")
    st.write(f"**Intent:** {intent}")
    st.write(f"**Sentiment:** {sentiment}")
    st.subheader("Context-Aware Response")
    st.write(f"**Suggested Action:** {rag_answer}")
    #st.subheader("Risk Classification")
    #st.write(f"**Risk Indicator (from query context):** {risk_level_from_query}")

    st.markdown("---")

    # Fraud risk (user enters transaction details)
    st.subheader("Fraud Risk Check (Transaction Details)")
    amount = st.number_input("Transaction Amount (₹)", value=1000.0)
    hour = st.slider("Hour of Day", 0, 23, 12)
    day = st.selectbox("Day of Week", fraud_le.classes_)
    is_international = st.selectbox("International?", ["Yes", "No"])
    velocity = st.selectbox("Velocity Flag?", ["Yes", "No"])
    geo = st.selectbox("Geo Anomaly?", ["Yes", "No"])
    high_amt = st.selectbox("High Amount?", ["Yes", "No"])

    if st.button("Check Fraud Risk"):
        sample = pd.DataFrame([{
            "amount_inr": amount,
            "hour_of_day": hour,
            "day_of_week": fraud_le.transform([day])[0],
            "is_international": 1 if is_international == "Yes" else 0,
            "velocity_flag": 1 if velocity == "Yes" else 0,
            "geo_anomaly_flag": 1 if geo == "Yes" else 0,
            "high_amount_flag": 1 if high_amt == "Yes" else 0
        }])
        sample_scaled = fraud_scaler.transform(sample)
        prob = fraud_model.predict_proba(sample_scaled)[0][1]
        st.write(f"Fraud probability: {prob:.4f}")
        st.write(f"Risk level: {map_risk(prob)}")