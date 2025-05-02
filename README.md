# 📚 Multilingual PDF RAG System with Chatbot (Hindi, English, Bengali, Chinese)

This project implements a **Multilingual Retrieval-Augmented Generation (RAG) System** capable of extracting, indexing, and retrieving information from scanned and digital PDFs in **Hindi, English, Bengali, and Chinese**. It uses **OCR**, **semantic chunking**, **dense embeddings**, and a **local LLM-based chatbot** to answer questions.


<img width="943" alt="Screenshot 2025-05-03 000457" src="https://github.com/user-attachments/assets/7fe4ad9e-aeb2-4b66-81ba-7e16e9fec4e4" />

---

## 📁 Folder Structure

```
📦Multilingual-PDF-RAG-System/
├── documents.py       # PDF extraction and OCR
├── chunker.py         # Smart text chunking
├── embeddings.py      # Embedding generation using multilingual models
├── retrieve.py        # Vector store creation and document retrieval
├── result1.py         # Question generation (data augmentation)
├── result2.py         # LLM-based question answering
├── chatbot.py         # Chatbot interface using Streamlit/Gradio
├── README.md          # Project overview and usage
└── requirements.txt   # Python dependencies
```

---

## 🚀 Features

- 🔤 **Multilingual PDF Support** (Hindi, English, Bengali, Chinese)
- 🧾 **Scanned PDF OCR** via Tesseract
- ✂️ **Recursive Chunking** for context-preserving splits
- 🧠 **Embeddings** using `sentence‑transformers/paraphrase‑MiniLM‑L6‑v2`
- 🔍 **FAISS Vector Store** for efficient retrieval
- 💬 **Chatbot Interface** with contextual memory
- 🧠 **Local LLM** inference (google/flan-t5-base)

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Multilingual-PDF-RAG-System.git
cd Multilingual-PDF-RAG-System
```

### 2. Create Virtual Environment

```bash
python -m venv rag_env
# Windows
rag_env\Scripts\activate
# Linux/macOS
source rag_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract (for OCR)

- [Download Tesseract](https://github.com/tesseract-ocr/tesseract)
- Add the Tesseract path to your system's environment variables.

---

## 🧩 Pipeline Overview

### 1. Document Parsing and OCR (`documents.py`)
- Extracts text from both scanned and digital PDFs.
- Applies OCR using Tesseract if needed.

### 2. Chunking (`chunker.py`)
- Uses recursive character splitting to chunk large texts into semantically meaningful blocks.

### 3. Embedding Creation (`embeddings.py`)
- Converts chunks into vector embeddings using a multilingual model.

### 4. Retrieval (`retrieve.py`)
- Saves indexed embeddings using FAISS.
- Performs hybrid search to fetch top relevant chunks.

### 5. Question Answering (`result2.py`)
- Answers questions using a local LLM like FLAN-T5 or TinyLLaMA.

### 6. Chatbot Interface (`chatbot.py`)
- Provides a conversational UI for asking questions about PDF documents.
- Supports chat history and memory.

---

## 💡 Example Usage

```bash
# 1. Parse and OCR PDFs
python documents.py

# 2. Chunk and embed
python chunker.py
python embeddings.py

# 3. Retrieve relevant chunks
python retrieve.py

# 4. Generate and answer questions
python result1.py
python result2.py

# 5. Launch chatbot
python chatbot.py
```

---

## 🧠 Model Notes

- 🧠 **Embedding Model**: `sentence-transformers/paraphrase-MiniLM-L6-v2` 
- 🧠 **LLM QA Model**: `google/flan-t5-small` 
Ensure models are either downloaded ahead or use `local_files_only=True`.

---

## 📌 To-Do

- [ ] Add Whisper for audio-to-text PDF generation
- [ ] Integrate CPU-based LLaMA.cpp model for ultra-low resource systems
- [ ] Streamlit UI for file upload + multilingual chat

---

## 🙋‍♂️ Author

**Vinay Kumar**  
AI/ML Engineer and Data Enthusiast | Focused on Multilingual NLP + LLM Ops

---

## 📄 License

MIT License – use freely with attribution.
