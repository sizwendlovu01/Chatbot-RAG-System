"""
retriever.py
------------
Builds a lightweight vector index over the Student Handbook chunks and
performs similarity search for a user's question.

Design note
~~~~~~~~~~~
Vercel's Python serverless functions have a limited deployment size and a
short cold-start budget, so heavy ML dependencies such as
`sentence-transformers` / `torch` are avoided. Instead we use scikit-learn's
TF-IDF vectorizer + cosine similarity as our "embedding" and vector search
layer. This is a well-established, fully-vectorized text-embedding
technique, needs no native binaries beyond numpy/scipy/scikit-learn, and is
easily swapped out for a hosted embeddings API later (see `USING_A_REAL_EMBEDDING_API`
note in README) without changing the public interface of this module.

The same `Retriever` class doubles as the "vector database": it stores
chunk text + metadata (source, page number) alongside the TF-IDF matrix
in memory, rebuilt once per cold start and then reused for the lifetime of
the serverless function instance (warm invocations).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .loader import load_and_chunk_handbook

# Minimum similarity score required before we trust a chunk enough to use
# it as context. Below this, we treat the question as "not answerable
# from the knowledge base".
DEFAULT_SIMILARITY_THRESHOLD = 0.09


@dataclass
class Retriever:
    chunk_size: int = 800
    overlap: int = 150
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD

    _records: List[Dict] = field(default_factory=list)
    _vectorizer: Optional[TfidfVectorizer] = None
    _matrix = None

    def build(self, pdf_path: Optional[str] = None) -> None:
        """Load the handbook, chunk it, and fit the TF-IDF index."""
        kwargs = {"chunk_size": self.chunk_size, "overlap": self.overlap}
        if pdf_path:
            kwargs["pdf_path"] = pdf_path
        self._records = load_and_chunk_handbook(**kwargs)

        corpus = [r["text"] for r in self._records]
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform(corpus)

    @property
    def is_built(self) -> bool:
        return self._vectorizer is not None and self._matrix is not None

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        """Return the top_k most relevant chunks for `query`.

        Each result dict contains: text, page, source, score.
        Returns an empty list if nothing clears the similarity threshold,
        signalling that the answer is not in the knowledge base.
        """
        if not self.is_built:
            self.build()

        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix).flatten()

        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score < self.similarity_threshold:
                continue
            record = self._records[idx]
            results.append(
                {
                    "text": record["text"],
                    "page": record["page"],
                    "source": record["source"],
                    "score": score,
                }
            )
        return results


if __name__ == "__main__":
    r = Retriever()
    r.build()
    for q in [
        "How much do the bootcamp fees cost?",
        "When is orientation day?",
        "What is the weather like today?",
    ]:
        print(f"\nQ: {q}")
        for res in r.search(q):
            print(f"  [p{res['page']} score={res['score']:.3f}] {res['text'][:120]}...")
