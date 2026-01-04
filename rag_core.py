# rag_core.py - Production Version
# Handles: PDF Loading, Embedding, Database, Ollama Connection, Quiz Generation

import os
import glob
import random
from typing import List, Tuple, Optional, Dict, Any

from pypdf import PdfReader
import chromadb
from chromadb.config import Settings

import requests
from sentence_transformers import SentenceTransformer

# === Configuration ===
DATA_DIR = "data_raw"
VECTOR_DIR = "vectorstore"
COLLECTION_NAME = "rl_docs"

# === Local Embeddings ===
# Downloads model automatically to ~/.cache/huggingface
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# === Local LLM (Ollama) ===
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"

# ---------------------------------------------------------
# 1. PDF LOADING & INDEXING
# ---------------------------------------------------------

def load_pdfs(data_dir: str = DATA_DIR) -> List[Tuple[str, str]]:
    """Reads all PDF files from the data directory."""
    docs = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    pattern = os.path.join(data_dir, "*.pdf")
    files = glob.glob(pattern)
    
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            
            if text.strip():
                docs.append((name, text))
                print(f"[PDF LOADED] {name}")
            else:
                print(f"[WARNING] Empty text in {name}")
                
        except Exception as e:
            print(f"[ERROR] Could not read {name}: {e}")
            
    return docs

def simple_chunk(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Splits text into overlapping chunks for better context."""
    text = text.replace("\r", " ").replace("\n", " ")
    chunks = []
    start = 0
    n = len(text)
    
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        
    return chunks

def get_chroma_collection():
    """Connects to the local vector database."""
    client = chromadb.PersistentClient(
        path=VECTOR_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(COLLECTION_NAME)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Converts text lists to vector lists."""
    return _embedder.encode(texts, convert_to_numpy=True).tolist()

def build_vector_index(docs: List[Tuple[str, str]]):
    """Rebuilds the entire database from scratch."""
    collection = get_chroma_collection()
    
    # Clear existing data to avoid duplicates
    try:
        existing = collection.get()
        if existing["ids"]:
            print(f"[INDEX] Deleting {len(existing['ids'])} old entries...")
            collection.delete(ids=existing["ids"])
    except Exception as e:
        print(f"[WARN] Could not clean database: {e}")

    all_chunks = []
    all_ids = []
    all_metas = []
    
    for doc_id, text in docs:
        chunks = simple_chunk(text)
        for i, chunk in enumerate(chunks):
            cid = f"{doc_id}::chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(cid)
            all_metas.append({"doc_id": doc_id})

    print(f"[INDEX] Embedding {len(all_chunks)} chunks... (This may take a moment)")
    
    if all_chunks:
        embeddings = embed_texts(all_chunks)
        collection.add(
            ids=all_ids,
            documents=all_chunks,
            metadatas=all_metas,
            embeddings=embeddings
        )
    print("[INDEX] Build Complete.")

# ---------------------------------------------------------
# 2. OLLAMA CONNECTION (ROBUST)
# ---------------------------------------------------------

def _call_ollama_chat(messages: List[dict], max_tokens: int = 500) -> str:
    """Sends messages to Ollama with robust error handling."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.3 
        }
    }
    
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
        
    except requests.exceptions.Timeout:
        return "Error: Ollama timed out. The model is taking too long to reply."
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Is 'ollama serve' running?"
    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------------------------------------------
# 3. RAG QUERY LOGIC
# ---------------------------------------------------------

def rag_query_with_history(question: str, history: List[dict], k: int = 5) -> Tuple[str, List[str]]:
    """
    Main RAG function. 
    Returns: (Answer String, List of Source Filenames)
    """
    collection = get_chroma_collection()
    q_emb = embed_texts([question])
    
    # 1. Retrieve relevant chunks
    results = collection.query(query_embeddings=q_emb, n_results=k)
    
    if not results["documents"] or not results["documents"][0]:
        return "I could not find any relevant information in your PDFs.", []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    
    # 2. Extract unique sources
    sources = list(set([m['doc_id'] for m in metas]))

    # 3. Prepare Context
    context_text = "\n\n".join([f"[{m['doc_id']}] {d}" for d, m in zip(docs, metas)])
    
    # 4. Construct Prompt
    system_prompt = (
        "You are a university teaching assistant. "
        "Answer the question strictly based on the Context provided below. "
        "If the answer is not in the context, state that you do not know. "
        "Cite the document names if possible."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history) # Append previous conversation
    messages.append({
        "role": "user", 
        "content": f"Context:\n{context_text}\n\nQuestion: {question}"
    })
    
    # 5. Generate Answer
    answer = _call_ollama_chat(messages)
    return answer, sources

# --- Wrapper for Backward Compatibility ---
def rag_query(question: str) -> str:
    """Legacy function that returns just the answer string."""
    answer, _ = rag_query_with_history(question, [])
    return answer

# ---------------------------------------------------------
# 4. NEW FEATURES (QUIZ)
# ---------------------------------------------------------

def generate_quiz_question() -> Dict[str, Any]:
    """
    Generates a multiple choice question and parses it into a dictionary.
    Returns: {'question': str, 'options': List[str], 'correct_answer': str}
    """
    collection = get_chroma_collection()
    existing = collection.get()
    
    if not existing["documents"]:
        return {"error": "No documents found. Please upload PDFs first."}

    # Pick 3 random chunks
    num_docs = len(existing["documents"])
    indices = random.sample(range(num_docs), min(3, num_docs))
    context = "\n".join([existing["documents"][i] for i in indices])

    # Strict prompt to ensure parsable output
    prompt = (
        "Based on the following context, generate a single multiple-choice question.\n"
        "Follow this EXACT format:\n"
        "Question: [The question text]\n"
        "Option A: [First option]\n"
        "Option B: [Second option]\n"
        "Option C: [Third option]\n"
        "Option D: [Fourth option]\n"
        "Answer: [Just the letter A, B, C, or D]\n\n"
        f"Context:\n{context}"
    )
    
    raw_text = _call_ollama_chat([{"role": "user", "content": prompt}])
    
    # Parse the response
    parsed = {"question": "", "options": [], "correct_answer": ""}
    lines = raw_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith("Question:"):
            parsed["question"] = line.replace("Question:", "").strip()
        elif line.startswith("Option"):
            # Keeps "Option A: Text" format for display
            parsed["options"].append(line)
        elif line.startswith("Answer:"):
            parsed["correct_answer"] = line.replace("Answer:", "").strip().upper()
            # Clean up if it says "Answer: Option A" -> "A"
            if "OPTION" in parsed["correct_answer"]:
                 parsed["correct_answer"] = parsed["correct_answer"].replace("OPTION", "").strip()
    
    # Fallback if parsing failed but we got text
    if not parsed["question"]:
        parsed["question"] = raw_text
        parsed["error_parsing"] = True
        
    return parsed