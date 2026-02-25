import os
import tempfile
import pandas as pd
import docx2txt
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # FAISS is faster than Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# ------------------------------------------
# Load Documents (Simple & Fast)
# ------------------------------------------
def load_documents(uploaded_files):
    """Load documents - no OCR, just text extraction"""
    docs = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name.lower()
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            print(f"📄 Processing: {uploaded_file.name}")
            
            # PDF - simple loader, no OCR
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(temp_file_path)
                pdf_docs = loader.load()
                for doc in pdf_docs:
                    doc.metadata["source"] = uploaded_file.name
                docs.extend(pdf_docs)
                print(f"   ✅ PDF: {len(pdf_docs)} pages")

            # Word
            elif file_name.endswith(".docx"):
                text = docx2txt.process(temp_file_path)
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": uploaded_file.name}
                    ))
                    print(f"   ✅ DOCX: {len(text)} chars")

            # Excel
            elif file_name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(temp_file_path)
                text = df.to_string(index=False)
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": uploaded_file.name}
                    ))
                    print(f"   ✅ Excel: {df.shape[0]} rows")

            # TXT
            elif file_name.endswith(".txt"):
                with open(temp_file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": uploaded_file.name}
                    ))
                    print(f"   ✅ TXT: {len(text)} chars")
    
    print(f"\n📊 Total: {len(docs)} pages/sections")
    return docs

# ------------------------------------------
# Split Documents (OPTIMAL CHUNK SIZE)
# ------------------------------------------
def split_documents(documents):
    """Split with optimal chunk size for speed"""
    
    print(f"✂️ Splitting {len(documents)} documents...")
    
    # ✅ OPTIMAL chunk size - larger = fewer chunks = faster
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,              # Optimal for financial docs
        chunk_overlap=150,              # Enough overlap
        length_function=len,
        separators=["\n\n", "\n", ".", " "]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# ------------------------------------------
# Create Embeddings (SMALL MODEL)
# ------------------------------------------
def create_embeddings():
    """Create embeddings with small model for speed"""
    
    # ✅ SMALLER MODEL - 80MB vs 440MB (5x faster)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Tiny but effective
        model_kwargs={'device': 'cpu'},
        encode_kwargs={
            'normalize_embeddings': True
            # No batch_size needed - FAISS handles batching automatically
        }
    )
    
    print("⚡ FAST MODE:")
    print("   ✅ Model: all-MiniLM-L6-v2 (80MB - 5x smaller)")
    print("   ✅ FAISS will handle automatic batching")
    return embeddings

# ------------------------------------------
# Build Vector Database (FAISS - Faster than Chroma)
# ------------------------------------------
def build_vector_db(uploaded_files, groq_api_key=None):
    """Main function - OPTIMIZED for speed"""
    
    # Load documents
    docs = load_documents(uploaded_files)
    if not docs:
        raise ValueError("No documents loaded")
    
    # Split into chunks (optimal size)
    chunks = split_documents(docs)
    
    # Create embeddings (small model)
    embeddings = create_embeddings()
    
    # ✅ FAISS.from_documents = AUTOMATIC BATCHING!
    # This single line does ALL the batching work internally
    print("💾 Creating FAISS vector store with automatic batching...")
    vectordb = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    print(f"✅ Done! {vectordb.index.ntotal} vectors")
    return vectordb

# ------------------------------------------
# Build RAG Chain
# ------------------------------------------
def build_rag_chain(vectordb, groq_api_key):
    
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )
    
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )
    
    prompt = PromptTemplate(
        template="""You are a Financial AI Assistant.

Context: {context}

Question: {question}

Answer (use exact numbers from context):""",
        input_variables=["context", "question"]
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa

# ------------------------------------------
# Query Function
# ------------------------------------------
def query_document(qa_chain, question):
    try:
        result = qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result.get("source_documents", [])
        }
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "source_documents": []}