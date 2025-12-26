# LeXiDesk Chatbot — quickstart

1. Create venv and install:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Ingest PDFs:
   python ingest/ingest.py path/to/your.pdf

3. Build embeddings:
   python embeddings/indexer.py

4. Run API:
   uvicorn api.main:app --reload --port 8000

5. Open UI:
   streamlit run ui/streamlit_app.py
