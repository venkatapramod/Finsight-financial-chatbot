import streamlit as st
from rag_pipeline import build_vector_db, build_rag_chain, query_document
import time

st.set_page_config(page_title="Financial RAG Chatbot", page_icon="💰", layout="centered")

st.title("💰 Financial RAG Chatbot")
st.write("Upload your financial documents (PDF / Word / Excel) and ask questions.")

# Show info about processing time
st.info("⏳ First time processing takes 1-2 minutes. Subsequent queries will be faster!")

# ------------------------------------------
# Hardcoded Groq API Key (Replace with your actual key)
# ------------------------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# ------------------------------------------
# Sidebar
# ------------------------------------------
with st.sidebar:
    st.header("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "xlsx", "xls"],
        accept_multiple_files=True
    )
    
    # Process documents button with PROGRESS BAR (Option 1)
    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            
            # Create progress bar
            progress_bar = st.progress(0, text="⏳ Starting document processing...")
            status_text = st.empty()
            
            try:
                # Step 1: Loading documents (10%)
                status_text.text("📄 Step 1/4: Loading and extracting text from documents...")
                progress_bar.progress(10)
                time.sleep(0.5)  # Small delay to show progress
                
                # Step 2: Building vector database with faster embeddings (20-80%)
                status_text.text("🧠 Step 2/4: Generating embeddings (this takes the longest)...")
                progress_bar.progress(30)
                
                # Build vector database (this does the heavy work)
                vectordb = build_vector_db(uploaded_files, GROQ_API_KEY)
                
                progress_bar.progress(80)
                
                # Step 3: Building RAG chain (80-95%)
                status_text.text("🔧 Step 3/4: Building RAG chain with Groq...")
                progress_bar.progress(90)
                qa_chain = build_rag_chain(vectordb, GROQ_API_KEY)
                
                # Step 4: Saving to session (95-100%)
                status_text.text("💾 Step 4/4: Saving to session...")
                progress_bar.progress(95)
                
                # Store in session state
                st.session_state['vectordb'] = vectordb
                st.session_state['qa_chain'] = qa_chain
                st.session_state['documents_processed'] = True
                
                # Complete!
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                time.sleep(0.5)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ Successfully processed {len(uploaded_files)} file(s)!")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error processing documents: {str(e)}")
    
    st.markdown("---")
    
    # Clear Chat Only Button (Option 2)
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        if 'messages' in st.session_state:
            st.session_state.messages = []
        st.success("Chat history cleared! Documents still loaded.")
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Upload financial documents (PDF/Word/Excel)
    2. Click 'Process Documents' (takes 1-2 mins)
    3. Ask questions about your documents
    4. Use 'Clear Chat' to start fresh conversation
    """)
    
    # Show processing status if documents are loaded
    if 'documents_processed' in st.session_state:
        st.markdown("---")
        st.markdown("✅ **Status:** Documents ready for questions")

# ------------------------------------------
# Main Chat Area
# ------------------------------------------
if 'documents_processed' in st.session_state and st.session_state['documents_processed']:
    
    # Header with Clear Chat button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.header("💬 Ask Questions")
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history if not exists
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(message["sources"][:2]):
                        source_name = doc.metadata.get('source', 'Unknown')
                        st.text(f"Source {i+1}: {source_name}")
                        st.text(doc.page_content[:200] + "...")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response with thinking indicator
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching documents and generating answer..."):
                try:
                    # Query the document
                    result = query_document(st.session_state['qa_chain'], prompt)
                    
                    response = result["answer"]
                    sources = result["source_documents"]
                    
                    st.markdown(response)
                    
                    # Show sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, doc in enumerate(sources[:2]):
                                source_name = doc.metadata.get('source', 'Unknown')
                                st.text(f"Source {i+1}: {source_name}")
                                st.text(doc.page_content[:200] + "...")
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

else:
    # Welcome message
    st.info("👆 Please upload your financial documents in the sidebar and click 'Process Documents' to get started!")
    
    # Sample questions
    with st.expander("📝 Sample Questions You Can Ask"):
        st.markdown("""
        - What is the total revenue?
        - What is the net income?
        - What are the total assets?
        - How much cash does the company have?
        - What are the main expenses?
        - Who is the CEO?
        """)

# ------------------------------------------
# Footer
# ------------------------------------------
st.markdown("---")
st.markdown("Built with LangChain, Groq, and Streamlit | Faster embeddings with batch processing")