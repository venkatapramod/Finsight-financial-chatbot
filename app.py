import streamlit as st
import os

from rag_pipeline import build_vector_db, build_rag_chain, query_document

st.set_page_config(page_title="Financial RAG Chatbot", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📁 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Choose files",
    type=["pdf", "docx", "xlsx", "txt"],
    accept_multiple_files=True
)

process_button = st.sidebar.button("🚀 Process Documents")

st.sidebar.markdown("---")
st.sidebar.button("🗑️ Clear Chat History")

st.sidebar.markdown("## Instructions")
st.sidebar.write("""
1. Upload financial documents  
2. Click 'Process Documents'  
3. Ask questions  
4. Clear chat anytime  
""")

# ---------------- MAIN UI ----------------
st.title("💰 Financial RAG Chatbot")
st.write("Upload your financial documents (PDF / Word / Excel) and ask questions.")

st.info("⏳ First time processing takes 1–2 minutes. Subsequent queries will be faster!")

# ---------------- SAFE SECRETS ----------------
HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not HF_TOKEN or not GROQ_API_KEY:
    st.error("❌ Missing API keys")
    st.stop()

# ---------------- SESSION STATE ----------------
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# ---------------- PROCESS DOCUMENTS ----------------
if process_button:
    if not uploaded_files:
        st.warning("Please upload files first")
    else:
        with st.spinner("Processing documents..."):
            vectordb = build_vector_db(uploaded_files, HF_TOKEN)
            st.session_state.rag_chain = build_rag_chain(vectordb, GROQ_API_KEY)
        st.success("✅ Documents processed successfully!")

# ---------------- ASK QUESTION ----------------
if st.session_state.rag_chain:
    query = st.text_input("Ask a financial question:")

    if query:
        with st.spinner("Thinking..."):
            result = query_document(st.session_state.rag_chain, query)

        st.write("### 📊 Answer:")
        st.write(result["answer"])