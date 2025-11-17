# build_index.py

from rag_core import load_pdfs, build_vector_index, DATA_DIR

def main():
    print(f"[BUILD] Loading PDFs from: {DATA_DIR}")
    docs = load_pdfs(DATA_DIR)
    print(f"[BUILD] Loaded {len(docs)} documents.")
    if not docs:
        print("[BUILD] No documents found. Check that your PDFs are in the 'data_raw' folder.")
        return
    build_vector_index(docs)

if __name__ == "__main__":
    main()
