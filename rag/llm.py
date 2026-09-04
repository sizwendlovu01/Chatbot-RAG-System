"""
llm.py
------
Thin wrapper around Groq's chat-completions endpoint
(OpenAI-compatible API: https://api.groq.com/openai/v1/chat/completions).

The assistant is instructed to answer strictly from the supplied context
(retrieved Student Handbook chunks) and to refuse otherwise.
"""

import os
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = (
    "You are the ZAIO Bootcamp Assistant. You answer questions about the "
    "Full-Stack AI Engineer Bootcamp using ONLY the context excerpts provided "
    "to you from the Student Handbook. "
    "Rules:\n"
    "1. Only use facts that appear in the provided context.\n"
    "2. Never invent, guess, or use outside knowledge.\n"
    "3. If the context does not contain the answer, respond with exactly this "
    "sentence and nothing else: \"{not_found}\"\n"
    "4. Keep answers concise and directly reference specific details (dates, "
    "fees, names, links) found in the context when relevant."
)


def _build_messages(question: str, context: str, not_found_msg: str):
    system = SYSTEM_PROMPT.format(not_found=not_found_msg)
    user = (
        f"Context from the Student Handbook:\n\"\"\"\n{context}\n\"\"\"\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def ask_grok(question: str, context: str, not_found_msg: str) -> str:
    """Call the Groq API with the retrieved context and return the answer text.

    (Function name kept as `ask_grok` so api/index.py doesn't need changes -
    it's just calling whichever LLM provider is wired up here.)

    If GROQ_API_KEY is not configured, or the request fails, we fall back to
    a safe default rather than raising, so the API never crashes on a
    missing/invalid key during grading/demo.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return (
            "GROQ_API_KEY is not configured on the server. "
            "Please set the GROQ_API_KEY environment variable."
        )

    if not context.strip():
        return not_found_msg

    payload = {
        "model": DEFAULT_MODEL,
        "messages": _build_messages(question, context, not_found_msg),
        "temperature": 0.1,
        "max_tokens": 500,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
        if not resp.ok:
            return f"Error contacting the Groq API: {resp.status_code} - {resp.text}"
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001 - surface a readable error to the API caller
        return f"Error contacting the Groq API: {exc}"