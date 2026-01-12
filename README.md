MATT Lucie 
AZZOUG Dalia 
Jeancy Candela Nisharize 
RUSTAMLI Sayqin 

# Agent-IA-academique

**Local, privacy-first RAG for students and researchers.** Upload PDFs, index them with Chroma, and query a local Ollama model through GUI, CLI, or REST API while keeping every document on your machine.

## Key Features
- **Full RAG pipeline:** Load lecture PDFs, chunk them, embed with `sentence-transformers`, and store them in a persistent Chroma collection for fast retrieval.
- **Privacy-first stack:** Ollama (LLM) + ChromaDB (vector store) run locally—no external inference or API keys required.
- **Multiple interfaces:** Streamlit web UI (with uploader, chat, and quiz), a lightweight CLI chat loop, and a FastAPI backend for integrations.
- **Dynamic quiz mode:** Generates multiple-choice questions from your indexed documents to help reinforce learning.
- **Source-aware answers:** Responses include citations referencing the originating PDFs.

## Tech Stack
- **LLM engine:** Ollama, defaulting to `llama3.2:3b`.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector database:** `chromadb` with a persistent `vectorstore/` directory.
- **PDF parsing:** `pypdf`.
- **Frontends:** `streamlit` (interactive UI) + CLI.
- **API:** `fastapi` with `uvicorn` for local serving.

## Installation
### Prerequisites
1. Install [Python 3.10+](https://www.python.org/downloads/) if not already available.
2. Install [Ollama](https://ollama.ai/docs/installation) and keep `ollama serve` running while using the agent.

### Steps
```bash
git clone https://github.com/your-username/Agent-IA-academique.git
cd Agent-IA-academique
python -m venv .venv
.venv\\Scripts\\Activate.ps1   # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
ollama pull llama3.2:3b
```

## Preparing Data
1. Place your lecture PDFs into the `data_raw/` directory.
2. Rebuild the embedding index:
   ```bash
   python build_index.py
   ```
   The script calls `load_pdfs` and `build_vector_index` from `rag_core.py`, recreating all chunks inside Chroma.

## Usage
### Streamlit Web UI (Recommended)
```bash
streamlit run rag_streamlit.py
```
- Upload PDFs, rebuild the index in-app, chat with the assistant, and explore the quiz tab. The UI shows Ollama’s status and lets you expand cited sources.

### CLI
```bash
python rag_cli.py
```
- Ask questions interactively; the session keeps a running history sent to Ollama for context.

### REST API
```bash
uvicorn rag_API:app --reload --port 8000
```
- POST to `/chat` with `question` + optional `history` (each entry is `{ role: str, content: str }`). Docs are at `http://127.0.0.1:8000/docs`.

## Project Structure
| File | Purpose |
| --- | --- |
| `rag_core.py` | Core RAG logic: PDF loading, chunking, embeddings, Chroma ops, Ollama calls, and quiz generation helpers. |
| `rag_streamlit.py` | Streamlit frontend with uploader, chat tab, quiz tab, and status indicators. |
| `rag_cli.py` | Terminal experience for quick Q&A with history. |
| `rag_API.py` | FastAPI backend exposing the RAG agent through `/chat`. |
| `build_index.py` | Script for rebuilding the index from the `data_raw/` folder. |
| `data_raw/` | Default storage for uploaded PDFs. |
| `vectorstore/` | Persistent Chroma database created at runtime. |

## Next Steps
1. Tune the Ollama prompt in `rag_core.py` or switch to another local model.
2. Add metadata (sections, timestamps, tags) to the Chroma documents.
3. Surface source snippets via the API or UI for higher trust.

---

## 🇫🇷 Version Française

**Agent-IA-Académique est une application RAG locale, confidentielle et conçue pour les étudiants et chercheurs.** Importez des PDF, posez des questions avec citations et générez des quiz sans envoyer vos données vers le cloud.

### Fonctionnalités principales
- Pipeline RAG complet : PDF → découpage → embeddings → Chroma.
- Confidentialité totale : Ollama + Chroma tournent localement.
- Interfaces multiples : Streamlit (upload/chat/quiz), CLI et API FastAPI.
- Quiz générés dynamiquement à partir des documents.
- Réponses sourcées avec noms de fichiers.

### Stack technique
- LLM : Ollama (modèle `llama3.2:3b`).
- Embeddings : `sentence-transformers/all-MiniLM-L6-v2`.
- Base vectorielle : `chromadb` (`vectorstore/` persistant).
- PDF : `pypdf`.
- Frontend : `streamlit`.
- Backend : `fastapi` avec `uvicorn`.

### Installation
1. Cloner le dépôt :
   ```bash
   git clone https://github.com/your-username/Agent-IA-academique.git
   cd Agent-IA-academique
   ```
2. Créer un environnement :
   ```bash
   python -m venv .venv
   .venv\\Scripts\\Activate.ps1   # Windows
   # source .venv/bin/activate     # macOS / Linux
   ```
3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ollama pull llama3.2:3b
   ```

### Préparer les PDFs
1. Déposez vos fichiers PDF dans `data_raw/`.
2. Reconstruisez l’index :
   ```bash
   python build_index.py
   ```

### Utilisation
- **Interface Streamlit** : `streamlit run rag_streamlit.py`. C’est l’expérience la plus complète (upload, chat, quiz, sources).
- **CLI** : `python rag_cli.py` pour un chat rapide dans le terminal.
- **API** : `uvicorn rag_API:app --reload --port 8000`. Postez vos questions sur `/chat` (docs : `http://127.0.0.1:8000/docs`).

### Structure du projet
| Fichier | Description |
| --- | --- |
| `rag_core.py` | Logique RAG : traitement PDF, chunking, embeddings, Chroma et Ollama. |
| `rag_streamlit.py` | Interface Streamlit avec uploader, chat, quiz et indicateurs d’état. |
| `rag_cli.py` | Boucle de chat en ligne de commande avec historique. |
| `rag_API.py` | API FastAPI exposant l’agent. |
| `build_index.py` | Script pour réindexer `data_raw/`. |
| `data_raw/` | Dossier de stockage par défaut pour les PDF. |
| `vectorstore/` | Base Chroma persistante générée dynamiquement. |

### Étapes suivantes
1. Adapter le prompt Ollama (`rag_core.py`) ou changer de modèle local.
2. Ajouter des métadonnées (Sections, dates, balises) aux documents Chroma.
3. Faire remonter l’extrait source via l’API/UI pour plus de transparence.
