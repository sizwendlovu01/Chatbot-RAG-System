# Part 4 — Test Cases

Knowledge base used: **Student Handbook (PDF) only**, per project scope for this
submission. The retrieval/indexing pipeline (`rag/loader.py`, `rag/retriever.py`)
is source-agnostic — a second source (e.g. the ZAIO website) can be added later
simply by producing the same `{text, page, source}` record shape and re-building
the index (see "Adding a second source" in `README.md`). The three "cannot be
answered" cases below stand in for both (a) the currently un-indexed website
source and (b) genuinely out-of-scope questions.

Each row shows: the question sent to `POST /ask`, the source the API returned,
and the answer produced by Grok using **only** the retrieved handbook context.

---

## ✅ Answerable from the Student Handbook

### 1. Fees
**Question:** "How much do the bootcamp fees cost and what is the payment deadline?"

**Retrieved Source:** `Student Handbook - Page 16`

**Generated Answer:**
> The total fee for the bootcamp is R 38,950. Fees must be paid in full before
> Orientation Day, which is scheduled for 15 January 2026. You can either pay
> upfront using Zaio's bank details (provided by your consultant), or apply for
> financing through Zaio's partners, Capitec and Manati.

---

### 2. Key dates
**Question:** "When is orientation day and how do I join?"

**Retrieved Source:** `Student Handbook - Page 7`

**Generated Answer:**
> Orientation Day is on 15 January 2026 at 10:00 AM (Harare/Pretoria time). It's
> hosted as a Zoom meeting titled "January Bootcamp Orientation" (Meeting ID:
> 823 7228 2183, Passcode: 119149). Before orientation, all communication happens
> by email; after the bootcamp starts, ongoing communication moves to Discord.

---

### 3. Live classes
**Question:** "What are the live class days and times?"

**Retrieved Source:** `Student Handbook - Page 11`

**Generated Answer:**
> Live classes run Tuesdays 9 am–11 am and Thursdays 6 pm–8 pm for the first 12
> weeks, after which they reduce to a single Tuesday 9 am–11 am session per week.
> All sessions are recorded, and tutors are on standby for students who can't
> attend live.

---

### 4. The team
**Question:** "Who is the lead instructor and how do I contact them?"

**Retrieved Source:** `Student Handbook - Page 15`

**Generated Answer:**
> The Lead Instructor is Kago Kola (kago@zaio.io). Other team contacts include
> Abi Faith for Admissions (abi@zaio.io), Asif Hassam for Technical Support
> (asif@zaio.io), and Suhana Patel for Student Success (suhana@zaio.io).

---

### 5. Hardware requirements
**Question:** "What are the minimum hardware requirements for my laptop?"

**Retrieved Source:** `Student Handbook - Page 5`

**Generated Answer:**
> You need a current Windows laptop with PowerShell or a Mac with Terminal, a
> dual-core Intel i5 / AMD 3000+ / Apple M1 processor or better, at least 4–8 GB
> RAM, a 256 GB+ SSD, and an internet connection of at least 10 Mbps download /
> 3 Mbps upload. If you're unsure what to buy, email asif@zaio.io for guidance.

---

### 6. Discord levelling system
**Question:** "What are the Discord levelling tags a student can earn, from Code Explorer to Code Mentor Pro Max?"

**Retrieved Source:** `Student Handbook - Page 21`

**Generated Answer:**
> The tags, in order, are: Lvl 1 Code Explorer, Lvl 2 Curious Coder, Lvl 4 Helper,
> Lvl 6 Team Player, Lvl 9 Mentor in Training, Lvl 12 Team Player Pro, Lvl 15 Code
> Mentor, Lvl 18 Team Player Ultra Pro Max ("a must hire"), Lvl 23 Code Mentor Pro,
> and Lvl 30 Code Mentor Pro Max. You hold only one tag at a time — it updates as
> you level up.

---

### 7. Grade breakdown
**Question:** "What is the grade breakdown for the final mark?"

**Retrieved Source:** `Student Handbook - Page 14`

**Generated Answer:**
> Final grades are made up of: Final Project (React + Node) 35%, Assignments
> 25%, Coding Challenges 25%, and Multiple-Choice Questions (MCQs) 15%. Coding
> challenges and MCQs are marked on the Zaio platform.

---

### 8. Placement & career
**Question:** "What career placement support does Zaio offer after the bootcamp, and what salary can I expect?"

**Retrieved Source:** `Student Handbook - Page 24`

**Generated Answer:**
> Zaio helps place students as Junior Developers or paid interns within its
> partner network — a final mark of 70%+ significantly improves your placement
> chances. Paid internships run 12 months at R4,000/month; full-time Junior
> Developer roles pay R10,000–R15,000/month. Zaio frames the first year as being
> about gaining experience, with salary growth expected from year two onward.

---

## 🚫 Not answerable from the knowledge base

### 9. Off-topic, everyday question
**Question:** "What's the weather like today?"

**Retrieved Source:** `N/A`

**Generated Answer:**
> I could not find that information in the available knowledge base.

---

### 10. Off-topic, general knowledge
**Question:** "What is the capital of France?"

**Retrieved Source:** `N/A`

**Generated Answer:**
> I could not find that information in the available knowledge base.

---

### 11. Plausible-sounding but not in the handbook
**Question:** "Does Zaio offer a part-time data science bootcamp?"

**Retrieved Source:** `N/A`

**Generated Answer:**
> I could not find that information in the available knowledge base.

*(This is the kind of question that, in the full two-source version of this
assignment, would be looked up on the ZAIO website rather than the handbook —
see "Adding a second source" in README.md.)*

---

## How these were produced

1. Each question was sent to `POST /ask` on the deployed API.
2. `rag/retriever.py` computed a TF-IDF similarity search over the 39 handbook
   chunks; results below the similarity threshold (0.09) are dropped, which is
   why questions 9–11 return zero retrieved chunks.
3. When chunks *are* retrieved, `rag/llm.py` sends them as context to Grok with
   a strict system prompt: only use the supplied context, and reply with the
   canonical refusal sentence if the answer isn't in it. This is why the
   refusal message is identical, word-for-word, every time it's needed.
