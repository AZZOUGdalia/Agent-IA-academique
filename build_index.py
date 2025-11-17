# build_index.py

from rag_core import load_pdfs, build_vector_index

def main():
    docs = load_pdfs()
    if not docs:
        print("No PDFs found in data_raw/. Put your RL/ML lecture PDFs there.")
        return
    build_vector_index(docs)

if __name__ == "__main__":
    main()
