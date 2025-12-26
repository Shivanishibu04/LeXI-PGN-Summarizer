# api/main.py
"""
FastAPI server: POST /qa {question}
Returns: answer, sources
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retrieval.retriever import retrieve
import os
import openai
from typing import List, Optional

app = FastAPI(title="LeXiDesk Chatbot API")

class QARequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

# Simple generator: prefer OPENAI if key set, else fall back to local HF in the notebook
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

@app.post("/qa")
def qa(req: QARequest):
    try:
        docs = retrieve(req.question, k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Build context
    context = "\n\n".join([f"[{d['doc_id']}:p{d['page']}] {d['text']}" for d in docs])

    # If OpenAI key present, call completion, else return context snippets (caller can use local model)
    if OPENAI_KEY:
        prompt = f"You are a legal assistant. Use ONLY the context to answer. If info not present say 'I cannot find...'.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{req.question}"
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if "gpt-4o-mini" in openai.Model.list().get("data",[]) else "gpt-4o",
            messages=[{"role":"user","content":prompt}],
            max_tokens=512, temperature=0.0
        )
        answer = resp["choices"][0]["message"]["content"]
    else:
        # no key: return context plus a short instruction for client to run local generator
        answer = "OpenAI API key not configured. Returning retrieval snippets in `sources`."
    return {"answer": answer, "sources": docs}

# health
@app.get("/health")
def health():
    return {"status":"ok"}
