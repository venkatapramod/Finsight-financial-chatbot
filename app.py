import streamlit as st
import os

from rag_pipeline import build_vector_db, build_rag_chain, query_document

st.set_page_config(page_title="Financial RAG Chatbot", layout="wide", page_icon="💰")

with st.sidebar:
    st.title("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "xlsx", "txt"],
        accept_multiple_files=True
    )

    process_button = st.button("🚀 Process Documents", use_container_width=True, type="primary")

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 📌 Instructions")
    st.markdown("""
    1. Upload financial documents  
    2. Click **Process Documents**  
    3. Ask questions below  
    4. Clear chat anytime  
    """)

    st.markdown("---")
    st.caption("Built with LangChain · Groq · Streamlit")

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not HF_TOKEN or not GROQ_API_KEY:
    st.error("❌ Missing API keys. Please set HF_TOKEN and GROQ_API_KEY.")
    st.stop()

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.title("💰 Financial RAG Chatbot")
st.write("Upload your financial documents (PDF / Word / Excel) and ask questions.")

if process_button:
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one file first.")
    else:
        with st.spinner("⏳ Processing documents... This may take 1-2 minutes."):
            vectordb = build_vector_db(uploaded_files, HF_TOKEN)
            st.session_state.rag_chain = build_rag_chain(vectordb, GROQ_API_KEY)
            st.session_state.chat_history = []  # reset chat on new doc
        st.success("✅ Documents processed! You can now ask questions.")


if st.session_state.rag_chain:

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["answer"])
            if chat["sources"]:
                with st.expander("📄 View Sources"):
                    for i, doc in enumerate(chat["sources"]):
                        source_name = doc.metadata.get("source", "Unknown")
                        st.caption(f"**Source {i+1}:** `{source_name}`")
                        st.caption(doc.page_content[:300] + "...")
                        if i < len(chat["sources"]) - 1:
                            st.markdown("---")

 
    question = st.chat_input("Ask a question about your documents...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = query_document(st.session_state.rag_chain, question)
            st.write(result["answer"])
            if result["source_documents"]:
                with st.expander("📄 View Sources"):
                    for i, doc in enumerate(result["source_documents"]):
                        source_name = doc.metadata.get("source", "Unknown")
                        st.caption(f"**Source {i+1}:** `{source_name}`")
                        st.caption(doc.page_content[:300] + "...")
                        if i < len(result["source_documents"]) - 1:
                            st.markdown("---")

      
        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["source_documents"]
        })

else:
  
    st.info("⏳ First time processing takes 1–2 minutes. Subsequent queries will be faster!")
    st.warning("👆 Please upload your financial documents in the sidebar and click **Process Documents** to get started!")

    with st.expander("📋 Sample Questions You Can Ask"):
        st.markdown("""
        - What was the total revenue in 2023?
        - What is the net income for Q3?
        - How did operating expenses change year over year?
        - What are the key risk factors mentioned?
        - Summarize the cash flow statement.
        """)
