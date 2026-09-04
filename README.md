# ZAIO Student Handbook — RAG Assistant

A Retrieval-Augmented Generation (RAG) API that answers questions about the
**Full-Stack AI Engineer Bootcamp** using the **Student Handbook PDF** as its
knowledge base, and [Grok (xAI)](https://x.ai) as the answer-generation LLM.

Built to be deployed as a serverless function on **Vercel**.

```
POST /ask
{ "question": "How much do the bootcamp fees cost?" }

→ 200 OK
{
  "answer": "The total fee for the bootcamp is R 38,950...",
  "source": "Student Handbook - Page 16"
}
```

---

## How it works (Parts 1–3 of the assignment)

```
handbook.pdf
     │  rag/loader.py
     │  - pypdf text extraction (+ fix for this PDF's per-character spacing)
     │  - strips page numbers / boilerplate / vertical decorative headings
     │  - splits into ~800-char overlapping chunks, tagged with page number
     ▼
[chunk, page, source] records
     │  rag/retriever.py
     │  - TF-IDF vectorization (this project's "embedding" — see note below)
     │  - stored in-memory as the vector index ("vector database")
     ▼
Retriever.search(question)
     │  - cosine similarity search
     │  - returns empty list if nothing clears the similarity threshold
     ▼
api/index.py  (POST /ask)
     │  - if no chunks retrieved → returns the refusal message immediately,
     │    source = "N/A" (no LLM call needed)
     │  - otherwise, retrieved chunks become the "context" for Grok:
     ▼
rag/llm.py → Grok API (xAI)
     │  - strict system prompt: "only use the provided context;
     │    otherwise reply with the exact refusal sentence"
     ▼
{ "answer": "...", "source": "Student Handbook - Page N" }
```

### Why TF-IDF instead of a neural embedding model?

`sentence-transformers` / `torch` are the "obvious" choice for embeddings, but
they add ~1–2 GB of dependencies, which blows past Vercel's serverless
function size limit and makes cold starts unacceptably slow. Instead this
project uses scikit-learn's `TfidfVectorizer` + cosine similarity as a
lightweight, dependency-friendly stand-in for the "generate embeddings /
store in a vector database" requirement:

- Chunks are vectorized once at (lazy) startup and cached in memory for the
  life of the warm serverless instance — this is the "vector database" layer.
- A new question is vectorized the same way and compared via cosine
  similarity — this is the "generate an embedding for the question / search"
  step.
- A similarity threshold (`DEFAULT_SIMILARITY_THRESHOLD` in
  `rag/retriever.py`) filters out chunks that aren't actually relevant, so
  clearly off-topic questions return **no** chunks at all.

The **Grok LLM call is the final safety net**: even if a borderline-relevant
chunk slips through retrieval, the system prompt in `rag/llm.py` instructs
Grok to answer *only* from the supplied context and to reply with the exact
refusal sentence otherwise. So refusal correctness doesn't depend solely on
the similarity threshold being perfectly tuned.

> **Swapping in real embeddings later:** if you want semantic (not just
> lexical) retrieval, replace `TfidfVectorizer`/`cosine_similarity` in
> `rag/retriever.py` with calls to any hosted embeddings API (OpenAI,
> Cohere, Voyage, etc.) — the rest of the pipeline (chunking, metadata,
> `/ask` endpoint, refusal logic) stays the same.

### Adding a second source (e.g. the ZAIO website)

The assignment brief asks for the Student Handbook **and** the ZAIO website.
This build uses **only the handbook**, per instructions for this submission —
but the pipeline was written so a second source is a drop-in addition:

1. Write a small scraper (e.g. `requests` + `BeautifulSoup`, or Puppeteer if
   you need JS-rendered pages) that fetches the target pages and strips
   nav/header/footer boilerplate.
2. Emit records in the exact same shape used by the PDF loader:
   `{"text": "...", "page": None, "source": "ZAIO Website", "url": "https://zaio.io/..."}`
3. Concatenate that list with `load_and_chunk_handbook()`'s output before
   calling `TfidfVectorizer.fit_transform(...)` in `Retriever.build()`.
4. Update the `source` string built in `api/index.py` to use `url` when the
   record's `source` is `"ZAIO Website"` instead of a page number.

No other code changes are required — retrieval, refusal, and the API
contract all stay the same.

---

## Project structure

```
zaio-rag-assistant/
├── api/
│   └── index.py          # FastAPI app + Vercel serverless entrypoint (POST /ask, GET /health)
├── rag/
│   ├── loader.py          # PDF extraction, cleaning, chunking
│   ├── retriever.py        # TF-IDF index + similarity search ("vector DB")
│   └── llm.py               # Grok (xAI) chat-completions wrapper
├── data/
│   └── handbook.pdf        # The Student Handbook (knowledge source)
├── public/
│   └── index.html            # Minimal browser UI for testing /ask
├── test_cases.md               # Part 4 — documented test cases
├── n8n_workflow.json             # Part 5 — n8n workflow (chat → API → response)
├── requirements.txt
├── vercel.json
└── .env.example
```

---

## Local setup

```bash
git clone <this-repo>
cd zaio-rag-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROK_API_KEY=sk-... (your xAI API key)

uvicorn api.index:app --reload
```

Then either open `public/index.html` in a browser (point `API_BASE` to
`http://localhost:8000` if serving it separately), or test directly:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How much do the bootcamp fees cost?"}'
```

```bash
curl http://localhost:8000/health
# {"status":"ok","chunks_indexed":39}
```

---

## Deploying to Vercel

1. Push this project to a GitHub repo.
2. In Vercel: **New Project → Import** the repo.
3. Vercel auto-detects `vercel.json` (Python builder pointed at
   `api/index.py`, with `data/` and `rag/` explicitly included via
   `includeFiles` so the PDF and the package are bundled into the function).
4. Under **Project Settings → Environment Variables**, add:
   | Key | Value |
   |---|---|
   | `GROK_API_KEY` | your xAI API key |
   | `GROK_MODEL` *(optional)* | e.g. `grok-4-fast` (defaults to this if unset) |
5. Deploy. Your API will be live at:
   - `https://<your-project>.vercel.app/ask` (POST)
   - `https://<your-project>.vercel.app/health` (GET)
   - `https://<your-project>.vercel.app/` (static test page, if you also
     deploy `public/index.html` as a static asset / rename it into your
     framework's public folder)

**No API key set?** The API won't crash — `/ask` will still retrieve the
correct source chunk and return a clear message telling you `GROK_API_KEY`
isn't configured, instead of throwing a 500 error. This is deliberate so
grading/demoing the retrieval layer doesn't require a live key.

---

## API reference

### `POST /ask`

**Request**
```json
{ "question": "When is orientation day?" }
```

**Response — answer found**
```json
{
  "answer": "Orientation Day is on 15 January 2026 at 10:00 AM...",
  "source": "Student Handbook - Page 7"
}
```

**Response — not in the knowledge base**
```json
{
  "answer": "I could not find that information in the available knowledge base.",
  "source": "N/A"
}
```

### `GET /health`
Returns `{"status": "ok", "chunks_indexed": <int>}` — useful to confirm the
PDF was parsed and indexed successfully after deployment.

---

## Part 5 — n8n integration

`n8n_workflow.json` can be imported directly into n8n (**Workflows → Import
from File**). It wires up:

1. **Chat Trigger** — gives you a shareable public chat URL out of the box.
2. **HTTP Request node** — calls `POST https://<your-vercel-app>/ask` with
   `{"question": <user message>}`. ⚠️ Update the URL in that node after you
   deploy.
3. **IF node** — branches on whether `source` came back as `"N/A"`.
4. **Set nodes** — format the reply (appending the 📄 source when one was
   found).
5. **Respond to Webhook** — sends the formatted reply back to the chat.
6. An optional, disabled **Slack node** is included as an example of
   forwarding responses to another destination (Discord, Telegram,
   WhatsApp, Microsoft Teams, and email nodes all follow the same pattern —
   just swap the node and point it at the same `{{ $json.reply }}`
   expression).

---

## Known limitations

- TF-IDF is a **lexical** similarity measure, so questions that use very
  different wording from the handbook (heavy paraphrasing) may retrieve
  weaker matches than a true semantic embedding model would. The Grok call
  still acts as a final check against hallucinating an answer regardless.
- The in-memory index is rebuilt on cold start. For a single ~2 MB PDF this
  is well under a second, so it hasn't been persisted to disk/blob storage,
  but that would be the next optimization for a much larger knowledge base.
#   C h a t b o t - R A G - S y s t e m  
 #   C h a t b o t - R A G - S y s t e m  
 