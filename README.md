# 📊 Finsight

**AI-Powered Financial Document Q&A Chatbot**  
Built with Retrieval-Augmented Generation (RAG)

---

## 🚀 Overview

**Finsight** is a Retrieval-Augmented Generation (RAG) based financial chatbot that allows users to upload financial documents (PDF, DOCX, XLSX, XLS) and ask natural language questions about their content.

Instead of generating generic answers, the system retrieves relevant document context from a vector database and produces grounded, document-backed responses — minimizing hallucinations and improving financial accuracy.

---

## 📷 Demo

**Live Demo:** [Try Finsight](https://finsight-financial-chatbot-fjcsiydyuhe5dkcgjitgpg.streamlit.app/)

![Finsight Screenshot](https://github.com/user-attachments/assets/b733f175-2ea9-4fdb-b6b9-b174caf8b738)
![Finsight Demo](https://github.com/user-attachments/assets/89936907-a634-4d69-904e-0d49d75eb115)

---

## 💡 Sample Questions

- "What was Apple's total revenue in 2023?"
- "What was iPhone revenue?"
- "What were R&D expenses?"

---

## 🎯 Use Cases

- Financial statement analysis
- Investor report review
- Balance sheet & income statement Q&A
- AI-powered financial document assistant
- Automated corporate report analysis

---

## ✨ Key Features

- Upload financial documents (PDF, DOCX, XLSX, XLS)
- Automatic text extraction and smart chunking
- Vector embeddings stored in ChromaDB
- Conversational financial Q&A interface
- Context-grounded responses
- Interactive Streamlit web application

---

## 🧠 Architecture

`User Query → Embedding Generation → Semantic Retrieval (ChromaDB) → Context Injection → Groq LLM Response`

This ensures answers are strictly based on uploaded document content.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq (ChatGroq)

---

## 📁 Project Structure

```
Finsight-financial-chatbot/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # RAG & embedding logic
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python runtime info
├── .streamlit/         # Streamlit configuration folder
└── .devcontainer/      # Development container
```

---

## 🔧 Technical Decisions

- **Embedding model:** sentence-transformers/all-MiniLM-L6-v2 — runs locally on CPU, no API costs; embeddings are normalized for better cosine similarity matching
- **Vector DB:** ChromaDB with local persistence (`./chroma_db`) — lightweight, no separate server needed for a Streamlit app
- **Chunking:** 500 characters with 100 overlap using RecursiveCharacterTextSplitter — smaller chunks improve retrieval precision for specific financial figures
- **Retrieval:** Top-k=8 semantic search — higher k gives the LLM more context to cross-reference financial line items across pages
- **LLM:** Groq's llama-3.1-8b-instant with temperature=0 — fast inference for chat UX, zero temperature for deterministic, factual financial answers
- **Prompt engineering:** Custom system prompt instructs the model to treat "net sales" / "total net sales" as equivalent to "revenue" — handles common terminology variation in 10-K filings

---

## 🏃 Run Locally

```
git clone https://github.com/venkatapramod/Finsight-financial-chatbot.git
cd Finsight-financial-chatbot
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
streamlit run app.py
```

---

## 📊 Evaluation

Manually tested on financial filings (Apple, Microsoft 10-Ks) with sample queries on revenue, expenses, and segment data. Formal evaluation pipeline planned — see Future Work.

---

## 🚀 Future Work

- Display source citations (filename + page) alongside each answer using existing metadata
- Add conversation memory for multi-turn follow-up questions
- Build automated evaluation pipeline with ground-truth Q&A pairs
- Implement re-ranking with cross-encoder for higher retrieval precision
- Add OCR support for scanned/image-based PDFs
- Table-aware chunking to preserve financial statement structure

---

## 📄 License

This project is licensed under the MIT License.
