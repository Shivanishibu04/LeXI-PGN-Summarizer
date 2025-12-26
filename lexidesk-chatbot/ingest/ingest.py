# lexidesk-chatbot/ingest/ingest.py
"""
Ingest PDFs -> page-level text -> sentence segmentation (using predict.segment_text)
Produces:
- lexidesk-chatbot/data/lexidesk_pages.jsonl
- lexidesk-chatbot/data/chunks.jsonl
"""

from pathlib import Path
import json
import fitz  # pymupdf
import sys
import importlib.util
import re

# --------------------------------------------------
# Resolve repo root correctly
# lexidesk-chatbot/ingest/ingest.py -> go up TWO levels
# --------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# Make repo importable (for src/, predict.py, etc.)
sys.path.insert(0, str(REPO_ROOT))

# Data directory (inside lexidesk-chatbot)
DATA_DIR = REPO_ROOT / "lexidesk-chatbot" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PAGES_JSONL = DATA_DIR / "lexidesk_pages.jsonl"
SAVED_CHUNKS = DATA_DIR / "chunks.jsonl"

# --------------------------------------------------
# Dynamically import predict.py from repo root
# --------------------------------------------------
PREDICT_PATH = REPO_ROOT / "predict.py"

print("Loading predict.py from:", PREDICT_PATH)

spec = importlib.util.spec_from_file_location("lexi_predict", str(PREDICT_PATH))
predict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict)

# --------------------------------------------------
# Choose model
# --------------------------------------------------
model = predict.hybrid_crf_model or predict.baseline_crf_model
use_hybrid = predict.hybrid_crf_model is not None

print(
    f"Model loaded | baseline={predict.baseline_crf_model is not None} "
    f"hybrid={predict.hybrid_crf_model is not None} cnn={predict.cnn_model is not None}"
)

# --------------------------------------------------
# PDF extraction
# --------------------------------------------------
def extract_pages_from_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        pages.append(
            {
                "doc_id": Path(pdf_path).name,
                "page": i + 1,
                "text": text,
            }
        )
    return pages

# --------------------------------------------------
# Sentence segmentation (SBD)
# --------------------------------------------------
def segment_pages(pages):
    seg_fn = predict.segment_text
    out = []

    for p in pages:
        try:
            sents = seg_fn(p["text"], model, use_hybrid_features=use_hybrid)
            if isinstance(sents, tuple):
                sents = sents[0]
        except Exception as e:
            print(f"[WARN] SBD failed on page {p['page']} – fallback used:", e)
            sents = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", p["text"])
                if s.strip()
            ]

        out.append(
            {
                "doc_id": p["doc_id"],
                "page": p["page"],
                "sentences": sents,
                "text": p["text"],
            }
        )

    return out

# --------------------------------------------------
# Chunking for RAG
# --------------------------------------------------
def chunk_pages(segmented_pages, max_chars=1600):
    chunks = []

    for p in segmented_pages:
        cur, cur_len = [], 0

        for s in p["sentences"]:
            if cur_len + len(s) > max_chars and cur:
                chunks.append(
                    {
                        "doc_id": p["doc_id"],
                        "page": p["page"],
                        "text": " ".join(cur),
                    }
                )
                cur = [s]
                cur_len = len(s)
            else:
                cur.append(s)
                cur_len += len(s)

        if cur:
            chunks.append(
                {
                    "doc_id": p["doc_id"],
                    "page": p["page"],
                    "text": " ".join(cur),
                }
            )

    return chunks

# --------------------------------------------------
# Main runner
# --------------------------------------------------
def run(pdf_paths):
    all_pages = []

    for pdf in pdf_paths:
        all_pages.extend(extract_pages_from_pdf(pdf))

    # Save page-level text
    with open(PAGES_JSONL, "w", encoding="utf-8") as f:
        for p in all_pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_pages)} pages -> {PAGES_JSONL}")

    # Sentence segmentation
    seg_pages = segment_pages(all_pages)

    # Chunking
    chunks = chunk_pages(seg_pages)

    with open(SAVED_CHUNKS, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Saved {len(chunks)} chunks -> {SAVED_CHUNKS}")

    return PAGES_JSONL, SAVED_CHUNKS


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDFs into LeXIDesk chatbot")
    parser.add_argument("pdfs", nargs="+", help="PDF files to ingest")
    args = parser.parse_args()

    run(args.pdfs)
