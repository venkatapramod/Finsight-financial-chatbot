import os
import tempfile
import pandas as pd
import docx2txt

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document



def load_documents(uploaded_files):
    docs = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name.lower()
            temp_path = os.path.join(temp_dir, uploaded_file.name)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

          
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
                pdf_docs = loader.load()
                for d in pdf_docs:
                    d.metadata["source"] = uploaded_file.name
                docs.extend(pdf_docs)

            
            elif file_name.endswith(".docx"):
                text = docx2txt.process(temp_path)
                docs.append(Document(
                    page_content=text,
                    metadata={"source": uploaded_file.name}
                ))

            
            elif file_name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(temp_path)
                text = df.to_string(index=False)
                docs.append(Document(
                    page_content=text,
                    metadata={"source": uploaded_file.name}
                ))

            
            elif file_name.endswith(".txt"):
                with open(temp_path, "r", encoding="utf-8") as f:
                    text = f.read()
                docs.append(Document(
                    page_content=text,
                    metadata={"source": uploaded_file.name}
                ))

    return docs



def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)



def create_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )



def build_vector_db(uploaded_files, hf_token=None):   # hf_token no longer needed
    docs = load_documents(uploaded_files)
    chunks = split_documents(docs)

   
    chunks = [c for c in chunks if c.page_content.strip()]

    embeddings = create_embeddings()   # local model, no API key

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectordb

def build_rag_chain(vectordb, groq_api_key):
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 8}
    )

    prompt = PromptTemplate(
        template="""
You are a highly accurate financial analysis assistant.

Use ONLY the provided document context to answer.
Do NOT guess beyond the given data.

When financial reports use the terms "net sales" or "total net sales",
treat them as equivalent to "revenue" unless explicitly separated.

Context:
{context}

Question:
{question}

Answer in a direct, factual manner:
""",
        input_variables=["context", "question"]
    )

    return {"llm": llm, "retriever": retriever, "prompt": prompt}



def query_document(rag_chain, question):
    llm = rag_chain["llm"]
    retriever = rag_chain["retriever"]
    prompt = rag_chain["prompt"]

    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])

    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)

    return {
        "answer": response.content,
        "source_documents": docs
    }
