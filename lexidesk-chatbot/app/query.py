# lexidesk-chatbot/app/query.py

import json
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INDEX_FILE = DATA_DIR / "faiss.index"
META_FILE = DATA_DIR / "chunks_meta.json"

def main():
    index = faiss.read_index(str(INDEX_FILE))
    meta = json.load(open(META_FILE, "r", encoding="utf-8"))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    while True:
        query = input("\nAsk a legal question (or 'exit'): ")
        if query.lower() == "exit":
            break

        q_emb = model.encode([query])
        D, I = index.search(q_emb, k=5)

        print("\nTop relevant passages:\n")
        for idx in I[0]:
            print("-", meta[idx]["text"][:500], "\n")

if __name__ == "__main__":
    main()
