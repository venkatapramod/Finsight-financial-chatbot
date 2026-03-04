# 📊 Finsight
**AI-Powered Financial Document Q&A Chatbot**  
Built with Retrieval-Augmented Generation (RAG)

---

## 🚀 Overview
**Finsight** is a Retrieval-Augmented Generation (RAG) based financial chatbot that allows users to upload financial documents (PDF, DOCX, XLSX, XLS) and ask natural language questions about their content.

Instead of generating generic answers, the system retrieves relevant document context from a vector database and produces grounded, document-backed responses — minimizing hallucinations and improving financial accuracy.

---
## 📷 Demo
<img width="1920" height="1080" alt="Screenshot " src="https://github.com/user-attachments/assets/b733f175-2ea9-4fdb-b6b9-b174caf8b738" />

<img width="1920" height="1080" alt="demo" src="https://github.com/user-attachments/assets/89936907-a634-4d69-904e-0d49d75eb115" />



---

## 💡 Sample Questions

- "What was Apple's total revenue in 2023?"
- "What was iPhone revenue?"
- "What were R&D expenses?"

---

## 🎯 Use Cases

- 📊 Financial statement analysis  
- 📈 Investor report review  
- 🏦 Balance sheet & income statement Q&A  
- 🤖 AI-powered financial document assistant  
- 📑 Automated corporate report analysis  

---

## ✨ Key Features

- 📂 Upload financial documents (PDF, DOCX, XLSX, XLS)  
- 🔎 Automatic text extraction and smart chunking  
- 🧠 Vector embeddings stored in ChromaDB  
- 💬 Conversational financial Q&A interface  
- 📌 Context-grounded responses  
- ⚡ Interactive Streamlit web application  

---

## 🧠 Architecture

User Query → Embedding Generation → Semantic Retrieval (ChromaDB) → Context Injection → Groq LLM Response  

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

```text
Finsight-financial-chatbot/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # RAG & embedding logic
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python runtime info (if needed)
├── .streamlit/         # Streamlit configuration folder
└── .devcontainer/      # Development container (optional)

📄 License
This project is licensed under the MIT License.

