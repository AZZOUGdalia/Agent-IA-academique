# rag_core.py

import os
import glob
from typing import List, Tuple
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# Configuration
DATA_DIR = "data_raw"
VECTOR_DIR = "vectorstore"
COLLECTION_NAME = "rl_docs"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

load_dotenv
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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


def get_chroma_collection(persist_dir: str = VECTOR_DIR, collection_name: str = COLLECTION_NAME):
    client_chroma = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client_chroma.get_or_create_collection(collection_name)
    return collection


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Use OpenAI embeddings."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [d.embedding for d in response.data]


def build_vector_index(docs: List[Tuple[str, str]]):
    """Build or rebuild the vector index from docs."""
    collection = get_chroma_collection()

    # Safe clear of existing data (compatible with new chromadb)
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


def rag_query(question: str, k: int = 5) -> str:
    """RAG pipeline: retrieve chunks, call LLM with context, return answer."""
    collection = load_vector_index()
    results = collection.query(
        query_texts=[question],
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

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


def rag_query_with_history(question: str, history: List[dict], k: int = 5) -> str:
    """
    Same as rag_query but with chat history.
    history = list of {"role": "user"/"assistant", "content": "..."}
    """
    collection = load_vector_index()
    results = collection.query(
        query_texts=[question],
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

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )
    return completion.choices[0].message.content
