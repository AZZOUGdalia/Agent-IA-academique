

# 🎓 Academic AI Agent (Agent IA Académique)

🇫🇷 **Français** | 🇬🇧 **English**

---

## 📌 Project Overview

The **Academic AI Agent** is an intelligent university assistant designed to help students **revise and understand their course materials** efficiently.

It leverages a **RAG (Retrieval-Augmented Generation)** architecture, enabling a **local AI model** to read PDF documents (lecture notes, textbooks) and answer questions **accurately with source citations**, ensuring transparency and reliability.

This project runs **entirely locally**, guaranteeing **data privacy**.

---

## ✨ Key Features

### 💬 Intelligent Chat

* Ask questions in natural language about your uploaded PDFs
* Context-aware answers based on document content

### 📚 Source Citations

* Each response clearly indicates **which document** was used

### 🧠 Quiz Generator

* Automatically generates **MCQs (Multiple Choice Questions)** from your courses
* Ideal for self-assessment and exam preparation

### ⚙️ Multiple Interfaces

* **Web Interface (Streamlit)** – Full visual experience
* **Command Line Interface (CLI)** – Fast testing
* **API (FastAPI)** – Easy integration into other applications

---

## 🛠️ Technical Prerequisites

Before running the project, make sure you have the following installed:

### 🔹 Python

* Version **3.8 or higher**

### 🔹 Ollama (Local AI Engine)

* Download from: **[https://ollama.com](https://ollama.com)**
* Pull the required model:

```bash
ollama pull llama3.2:3b
```

⚠️ **Important:** Keep the Ollama application running in the background while using the agent.

---

## 🚀 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/AZZOUGdalia/Agent-IA-academique.git
cd Agent-IA-academique
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Getting Started

### 🌐 Web Interface (Recommended)

Run the following command:

```bash
python -m streamlit run rag_streamlit.py
```

* The app will open automatically at:
  **[http://localhost:8501](http://localhost:8501)**
* Upload your PDFs via the sidebar
* Click **"Process & Index Files"**
* Start chatting with your academic AI assistant

---

### 💻 Other Interfaces

#### 🔹 Command Line Interface (CLI)

```bash
python rag_cli.py
```

#### 🔹 API (FastAPI)

```bash
uvicorn rag_API:app --reload
```

* API Documentation available at:
  **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## 📂 Project Structure

```text
Agent-IA-academique/
│
├── rag_core.py        # Core logic (PDF loading, ChromaDB, Ollama)
├── rag_streamlit.py   # Web interface (Streamlit)
├── rag_API.py         # FastAPI backend
├── rag_cli.py         # Command-line interface
│
├── data_raw/          # Temporary storage for uploaded PDFs
├── vectorstore/       # Vector database (ChromaDB)
├── requirements.txt
└── README.md
```

---

## ❓ Troubleshooting

| Issue                    | Solution                                       |
| ------------------------ | ---------------------------------------------- |
| **AI Engine Offline**    | Run `ollama serve`                             |
| **Missing module error** | Run `pip install -r requirements.txt`          |
| **Model not responding** | Verify download with `ollama pull llama3.2:3b` |

---

## 🎯 Use Cases

* University exam revision
* Understanding complex lecture materials
* Creating quizzes automatically from notes
* Secure, offline academic AI assistant

---

## 📜 License & Credits

Developed as an **academic AI assistant project** using:

* **Ollama**
* **ChromaDB**
* **Streamlit**
* **FastAPI**


