# embeddings/indexer.py
"""
Embed chunks and build FAISS index
Saves: index/faiss.index and index/metadata.pkl
"""
from pathlib import Path
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
INDEX_DIR = REPO_ROOT / "index"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = [json.loads(l) for l in f]
    return chunks

def build_index(model_name="all-MiniLM-L6-v2"):
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    embed_model = SentenceTransformer(model_name)
    embs = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs.astype("float32"))
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("Index built:", index.ntotal, "vectors")

if __name__ == "__main__":
    build_index()
