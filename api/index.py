"""
api/index.py
------------
FastAPI application exposing the RAG assistant. This file is the Vercel
serverless entrypoint (see vercel.json) — Vercel's Python runtime looks
for a module-level `app` (ASGI application) in this file.

It can also be run locally with:
    uvicorn api.index:app --reload
"""

import os
import sys

# Make the project root importable (so `rag.*` resolves both locally and
# inside the Vercel serverless function).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag.retriever import Retriever
from rag.llm import ask_grok
from api.ui import CHAT_UI_HTML

NOT_FOUND_MSG = "I could not find that information in the available knowledge base."

app = FastAPI(
    title="ZAIO Student Handbook RAG Assistant",
    description=(
        "Retrieval-Augmented Generation API answering questions about the "
        "Full-Stack AI Engineer Bootcamp using the Student Handbook as its "
        "only knowledge source."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Built lazily on first request, then cached for the lifetime of the warm
# serverless instance (avoids re-parsing the PDF on every request).
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
        _retriever.build()
    return _retriever


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")


class AskResponse(BaseModel):
    answer: str
    source: str


@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the chat UI (embedded HTML, see api/ui.py) at the root URL."""
    return HTMLResponse(content=CHAT_UI_HTML)


@app.get("/api/status")
def api_status():
    return {
        "status": "ok",
        "service": "ZAIO Student Handbook RAG Assistant",
        "usage": "POST /ask  { \"question\": \"...\" }",
    }


@app.get("/health")
def health():
    try:
        retriever = get_retriever()
        return {"status": "ok", "chunks_indexed": len(retriever._records)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    retriever = get_retriever()
    results = retriever.search(question, top_k=4)

    if not results:
        return AskResponse(answer=NOT_FOUND_MSG, source="N/A")

    context = "\n\n".join(f"[Page {r['page']}] {r['text']}" for r in results)
    top_result = results[0]
    source = f"Student Handbook - Page {top_result['page']}"

    answer = ask_grok(question, context, NOT_FOUND_MSG)

    # Safety net: if the model still refused despite having context, make
    # sure the source reported matches the refusal (no source to cite).
    if answer.strip() == NOT_FOUND_MSG:
        source = "N/A"

    return AskResponse(answer=answer, source=source)