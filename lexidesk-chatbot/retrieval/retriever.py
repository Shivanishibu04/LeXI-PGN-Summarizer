# retrieval/retriever.py
from pathlib import Path
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = REPO_ROOT / "index"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
INDEX = faiss.read_index(str(INDEX_DIR / "faiss.index"))
with open(INDEX_DIR / "metadata.pkl", "rb") as f:
    METADATA = pickle.load(f)

def retrieve(query, k=5):
    q_emb = EMBED_MODEL.encode([query]).astype("float32")
    D,I = INDEX.search(q_emb, k)
    results=[]
    for idx,dist in zip(I[0], D[0]):
        m = METADATA[idx].copy()
        m["distance"] = float(dist)
        m["chunk_id"] = int(idx)
        results.append(m)
    return results
