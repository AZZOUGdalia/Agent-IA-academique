# rag_core.py - version sans OpenAI, avec embeddings locaux + Ollama

import os
import glob
from typing import List, Tuple

from pypdf import PdfReader
import chromadb
from chromadb.config import Settings

import requests
from sentence_transformers import SentenceTransformer

# === Config générale ===
DATA_DIR = "data_raw"
VECTOR_DIR = "vectorstore"
COLLECTION_NAME = "rl_docs"

# === Config embeddings (gratuit, local) ===
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# === Config Ollama (LLM gratuit, local) ===
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3"  # à récupérer avec: ollama pull llama3


def load_pdfs(data_dir: str = DATA_DIR) -> List[Tuple[str, str]]:
    """Load all PDF files and return list of (doc_id, text)."""
    docs = []
    pattern = os.path.join(data_dir, "*.pdf")
    for path in glob.glob(pattern):
        name = os.path.splitext(os.path.basename(path))[0]
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        if text.strip():
            docs.append((name, text))
            print(f"[PDF loaded] {name}")
        else:
            print(f"[WARNING] No text extracted from {name}")
    return docs


def simple_chunk(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Simple character-based chunking."""
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


def get_chroma_collection(
    persist_dir: str = VECTOR_DIR,
    collection_name: str = COLLECTION_NAME
):
    """Create or load a persistent Chroma collection."""
    client_chroma = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client_chroma.get_or_create_collection(collection_name)
    return collection


# ============= Embeddings (LOCAL, GRATUIT) =============

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embeddings locaux avec SentenceTransformers.
    Retourne une liste de vecteurs de floats.
    """
    embeddings = _embedder.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


# ============= Construction de l'index vecteur =============

def build_vector_index(docs: List[Tuple[str, str]]):
    """Build or rebuild the vector index from docs."""
    collection = get_chroma_collection()

    # Clear existing data
    try:
        existing = collection.get()
        ids = existing.get("ids", [])
        if ids:
            print(f"[INDEX] Deleting {len(ids)} existing embeddings...")
            collection.delete(ids=ids)
    except Exception as e:
        print(f"[WARN] Could not clear collection: {e}")

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for doc_id, text in docs:
        chunks = simple_chunk(text)
        for i, chunk in enumerate(chunks):
            cid = f"{doc_id}::chunk_{i}"
            meta = {"doc_id": doc_id, "chunk_index": i}
            all_chunks.append(chunk)
            all_ids.append(cid)
            all_metadatas.append(meta)

    print(f"[INDEX] Embedding {len(all_chunks)} chunks...")
    embeddings = embed_texts(all_chunks)

    collection.add(
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metadatas,
        embeddings=embeddings,
    )
    print("[INDEX] Done.")
    return collection


def load_vector_index():
    """Load existing Chroma collection."""
    return get_chroma_collection()


# ============= LLM via OLLAMA (LOCAL, GRATUIT) =============

def _call_ollama_chat(messages: List[dict]) -> str:
    """
    Appelle le modèle local via Ollama (format style ChatGPT).
    messages = [{"role": "system"/"user"/"assistant", "content": "..."}]
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,  # plus simple à gérer que le streaming
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Format standard d'Ollama: { "message": { "role": "...", "content": "..." }, ... }
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]

    # fallback si jamais la structure change
    return str(data)


# ============= RAG simple =============

def rag_query(question: str, k: int = 5) -> str:
    """RAG pipeline: retrieve chunks, call LLM with context, return answer."""
    collection = load_vector_index()
        # on embed la question avec notre SentenceTransformer
    q_emb = embed_texts([question])  # -> liste de 1 vecteur

    results = collection.query(
        query_embeddings=q_emb,
        n_results=k
    )


    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_blocks = []
    for doc, meta in zip(docs, metas):
        context_blocks.append(f"[{meta['doc_id']}] {doc}")

    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an academic assistant specialized in Reinforcement Learning and Machine Learning. "
        "You answer in clear, structured English. "
        "Use only the information from the provided context. If something is not in the context, say you don't know."
    )

    user_prompt = (
        f"Context (excerpts from RL/ML lecture notes):\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in English. If you need equations, describe them in plain text."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return _call_ollama_chat(messages)


def rag_query_with_history(question: str, history: List[dict], k: int = 5) -> str:
    """
    Same as rag_query but with chat history.
    history = list of {"role": "user"/"assistant", "content": "..."}
    """
    collection = load_vector_index()
        q_emb = embed_texts([question])
    results = collection.query(
        query_embeddings=q_emb,
        n_results=k
    )


    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_blocks = []
    for doc, meta in zip(docs, metas):
        context_blocks.append(f"[{meta['doc_id']}] {doc}")

    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an academic assistant specialized in Reinforcement Learning and Machine Learning. "
        "You answer in clear, structured English. "
        "Use only the information from the provided context. If something is not in the context, say you don't know."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    )

    return _call_ollama_chat(messages)
