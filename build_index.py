from sentence_transformers import SentenceTransformer

# modèle d'embedding gratuit et local
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    """
    texts : liste de strings
    return : liste de vecteurs (list[list[float]])
    """
    embeddings = _embedder.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()
