"""
loader.py
---------
Loads the Student Handbook PDF, cleans extracted text (strips repeated
headers/footers/page-number artefacts) and splits it into overlapping
chunks ready for embedding / TF-IDF indexing.

Each chunk is returned as a dict:
    {
        "text": "...",
        "page": 3,            # 1-indexed PDF page number
        "source": "Student Handbook"
    }
"""

import os
import re
from typing import List, Dict

from pypdf import PdfReader

DEFAULT_PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "handbook.pdf")

# Lines that are pure page-number artefacts, e.g. "0 3", "1 5", "12"
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,2}\s*\d?\s*$")
# Common repeated boilerplate found in this handbook's footer/header
_BOILERPLATE_PATTERNS = [
    re.compile(r"^Z\s*A\s*I\s*O$", re.IGNORECASE),
]


def _fix_char_spaced_text(raw_text: str) -> str:
    """Repair PDFs (like this handbook) whose font/encoding makes pypdf
    extract text with a single space between every character, e.g.
    "T o  e f f e c t i v e l y" instead of "To effectively".

    Heuristic: in the broken extraction, real word boundaries are marked
    by a DOUBLE space while individual characters within a word are
    separated by a SINGLE space. We detect this pattern (a high ratio of
    single-character tokens) and, only then, collapse it back into normal
    words by treating double-spaces as word boundaries and removing the
    single spaces in between.
    """
    tokens = raw_text.split(" ")
    if not tokens:
        return raw_text
    single_char_ratio = sum(1 for t in tokens if len(t) == 1) / len(tokens)
    if single_char_ratio < 0.5:
        # Looks like normal extraction already - leave it alone.
        return raw_text

    fixed = raw_text.replace("  ", "\x00")  # mark real word boundaries
    fixed = fixed.replace(" ", "")  # collapse intra-word char spacing
    fixed = fixed.replace("\x00", " ")  # restore word boundaries
    return fixed


def _clean_page_text(raw_text: str) -> str:
    """Remove page numbers, isolated single-letter/vertical-title lines,
    and collapse excess whitespace from a single PDF page's extracted text."""
    raw_text = _fix_char_spaced_text(raw_text)
    lines = raw_text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _PAGE_NUM_RE.match(stripped):
            continue
        if any(p.match(stripped) for p in _BOILERPLATE_PATTERNS):
            continue
        # Skip single-character "vertical title" artefacts (e.g. large
        # decorative letters spelling "CONTENTS" down the page)
        if len(stripped) == 1 and not stripped.isdigit():
            continue
        cleaned_lines.append(stripped)
    text = " ".join(cleaned_lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_pdf_pages(pdf_path: str = DEFAULT_PDF_PATH) -> List[Dict]:
    """Extract cleaned text per page from the PDF.

    Returns a list of {"page": int, "text": str} for every non-empty page.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = _clean_page_text(raw)
        if cleaned:
            pages.append({"page": i, "text": cleaned})
    return pages


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks by character count, breaking on
    sentence boundaries where possible so chunks stay coherent."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # try to break on a sentence boundary near the end
        if end < text_len:
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size * 0.5:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_and_chunk_handbook(
    pdf_path: str = DEFAULT_PDF_PATH,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Dict]:
    """Full pipeline: load PDF -> clean per page -> chunk -> attach metadata.

    Returns a list of dicts: {"text": str, "page": int, "source": "Student Handbook"}
    """
    # Chunks shorter than this are dropped: very short fragments (e.g. a
    # cover-page title) carry little information yet, under TF-IDF's L2
    # normalization, can dominate similarity scores purely because they
    # have few terms diluting a single keyword match. Filtering them out
    # keeps retrieval focused on substantive content.
    MIN_CHUNK_CHARS = 60

    pages = load_pdf_pages(pdf_path)
    records = []
    for page in pages:
        for chunk in chunk_text(page["text"], chunk_size=chunk_size, overlap=overlap):
            if len(chunk) < MIN_CHUNK_CHARS:
                continue
            records.append(
                {
                    "text": chunk,
                    "page": page["page"],
                    "source": "Student Handbook",
                }
            )
    return records


if __name__ == "__main__":
    recs = load_and_chunk_handbook()
    print(f"Loaded {len(recs)} chunks from handbook.")
    for r in recs[:3]:
        print(f"--- page {r['page']} ---")
        print(r["text"][:200])
