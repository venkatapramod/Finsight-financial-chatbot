import os
import tempfile
import pandas as pd
import docx2txt
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# -------------------------------------------------------------------
# Load Documents
# -------------------------------------------------------------------
def load_documents(uploaded_files):
    docs = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name.lower()
            temp_path = os.path.join(temp_dir, uploaded_file.name)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            print(f"📄 Processing: {uploaded_file.name}")

            # PDF
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
                pdf_docs = loader.load()
                for d in pdf_docs:
                    d.metadata["source"] = uploaded_file.name
                docs.extend(pdf_docs)
                print(f"   ✅ PDF: {len(pdf_docs)} pages")

            # DOCX
            elif file_name.endswith(".docx"):
                text = docx2txt.process(temp_path)
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": uploaded_file.name}
                    ))
                print(f"   ✅ DOCX loaded")

            # Excel
            elif file_name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(temp_path)
                text = df.to_string(index=False)
                docs.append(Document(
                    page_content=text,
                    metadata={"source": uploaded_file.name}
                ))
                print(f"   ✅ Excel: {df.shape[0]} rows")

            # TXT
            elif file_name.endswith(".txt"):
                with open(temp_path, "r", encoding="utf-8") as f:
                    text = f.read()
                docs.append(Document(
                    page_content=text,
                    metadata={"source": uploaded_file.name}
                ))
                print(f"   ✅ TXT loaded")

    print(f"📊 Total sections: {len(docs)}")
    return docs


# -------------------------------------------------------------------
# Split Documents
# -------------------------------------------------------------------
def split_documents(documents):
    print(f"✂️ Splitting {len(documents)} documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


# -------------------------------------------------------------------
# Embeddings
# -------------------------------------------------------------------
def create_embeddings():
    print("⚡ Using Embeddings: all-MiniLM-L6-v2")
    embed = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embed


# -------------------------------------------------------------------
# Build Chroma Vector DB
# -------------------------------------------------------------------
def build_vector_db(uploaded_files, groq_api_key=None):
    docs = load_documents(uploaded_files)
    if not docs:
        raise ValueError("No documents loaded")

    chunks = split_documents(docs)
    embeddings = create_embeddings()

    print("💾 Creating Chroma vector store...")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="financial_docs"
    )

    print("✅ Vector DB Ready!")
    return vectordb


# -------------------------------------------------------------------
# Build RAG Chain
# -------------------------------------------------------------------
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

Answer using ONLY the information from the context.
If the context does not contain the answer, say 'Not enough information'.
""",
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


# -------------------------------------------------------------------
# Query
# -------------------------------------------------------------------
def query_document(qa_chain, question):
    try:
        result = qa_chain.invoke({"query": question})
        return {
            "answer": result.get("result", "No answer"),
            "source_documents": result.get("source_documents", [])
        }
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "source_documents": []}