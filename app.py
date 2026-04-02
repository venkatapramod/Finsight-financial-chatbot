import streamlit as st
import os

from rag_pipeline import build_vector_db, build_rag_chain, query_document

st.set_page_config(page_title="Financial RAG Chatbot")

st.title("💰 Financial RAG Chatbot")
st.write("Upload your financial documents (PDF / Word / Excel) and ask questions.")

# ✅ SAFE SECRET HANDLING (FIXED)
HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not HF_TOKEN or not GROQ_API_KEY:
    st.error("❌ Missing API keys. Please add HF_TOKEN and GROQ_API_KEY in Streamlit secrets.")
    st.stop()

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "xlsx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info("⏳ First time processing takes 1–2 minutes...")

    vectordb = build_vector_db(uploaded_files, HF_TOKEN)
    rag_chain = build_rag_chain(vectordb, GROQ_API_KEY)

    query = st.text_input("Ask a financial question:")

    if query:
        with st.spinner("Thinking..."):
            result = query_document(rag_chain, query)

        st.success("Answer:")
        st.write(result["answer"])