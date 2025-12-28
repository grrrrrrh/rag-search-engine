from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

SCORE_PRECISION = 3



# ---------- paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../cli/lib -> parents[2] == repo root
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DOC_EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"
CHUNK_EMBEDDINGS_PATH = CACHE_DIR / "chunk_embeddings.npy"
CHUNK_METADATA_PATH = CACHE_DIR / "chunk_metadata.json"


# ---------- data ----------
def load_movies() -> list[dict[str, Any]]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return data["movies"]


# ---------- chunking ----------
def semantic_chunk_text(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    """Split text into sentence-based chunks with sentence overlap."""
    text = (text or "").strip()
    if not text:
        return []

    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_chunk_size:
        raise ValueError("overlap must be smaller than max_chunk_size")

    # Split on sentence boundaries
    raw = re.split(r"(?<=[.!?])\s+", text)

    # Edge case: single "sentence" with no terminal punctuation => keep whole text as one sentence
    if len(raw) == 1 and (not text.endswith((".", "!", "?"))):
        sentences = [text]
    else:
        sentences = [t.strip() for t in raw]
        sentences = [t for t in sentences if t]

    if not sentences:
        return []

    # Short text => one chunk
    if len(sentences) <= max_chunk_size:
        chunk = " ".join(sentences).strip()
        return [chunk] if chunk else []

    step = max_chunk_size - overlap
    chunks: list[str] = []
    i = 0

    # IMPORTANT: keep tail only if it contains MORE than overlap (prevents tiny redundant chunks)
    while True:
        chunk = " ".join(sentences[i : i + max_chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

        i += step
        if i >= len(sentences) or (len(sentences) - i) <= overlap:
            break

    return chunks

# ---------- semantic search ----------
class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.documents: list[dict[str, Any]] = []
        self.document_map: dict[int, dict[str, Any]] = {}

        self.embeddings: np.ndarray | None = None

    def generate_embedding(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            raise ValueError("Text must be non-empty.")
        emb = self.model.encode(text)
        return np.asarray(emb)

    def build_embeddings(self, documents: list[dict[str, Any]]) -> np.ndarray:
        self.documents = documents
        self.document_map = {int(d["id"]): d for d in documents}

        # Embed descriptions (or title fallback)
        texts = []
        for d in documents:
            desc = (d.get("description") or "").strip()
            if not desc:
                desc = (d.get("title") or "").strip()
            texts.append(desc)

        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        self.embeddings = np.asarray(self.embeddings)

        np.save(DOC_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict[str, Any]]) -> np.ndarray:
        self.documents = documents
        self.document_map = {int(d["id"]): d for d in documents}

        if DOC_EMBEDDINGS_PATH.exists():
            self.embeddings = np.load(DOC_EMBEDDINGS_PATH)
            return self.embeddings

        return self.build_embeddings(documents)


# ---------- commands used by CLI/tests ----------
def verify_model() -> None:
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text: str) -> None:
    ss = SemanticSearch()
    emb = ss.generate_embedding(text)
    a, b, c = float(emb[0]), float(emb[1]), float(emb[2])
    # Print enough precision so substrings like "-0.035" can match reliably
    print(f"First 3 dimensions: {a:.6f} {b:.6f} {c:.6f}")
    print(f"Dimensions: {emb.shape[0]}")


def verify_embeddings() -> None:
    movies = load_movies()
    ss = SemanticSearch()
    embs = ss.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embs.shape[0]} vectors in {embs.shape[1]} dimensions")


def embed_query_text(query: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)

    # EXACT formatting boot.dev expects (no commas, 3 decimals)
    first5 = embedding[:5].tolist()
    formatted = " ".join(f"{x:.3f}" for x in first5)

    print(f"Query: {query}")
    print(f"First 5 dimensions: [{formatted}]")
    print(f"Shape: {embedding.shape}")


# ---------- chunked semantic embeddings ----------
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_metadata: list[dict[str, int]] | None = None

    def build_chunk_embeddings(self, documents: list[dict[str, Any]]) -> np.ndarray:
        self.documents = documents
        self.document_map = {int(d["id"]): d for d in documents}

        all_chunks: list[str] = []
        chunk_metadata: list[dict[str, int]] = []

        for movie_idx, doc in enumerate(documents):
            desc = (doc.get("description") or "").strip()
            if not desc:
                continue

            chunks = semantic_chunk_text(desc, max_chunk_size=4, overlap=1)
            total = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {"movie_idx": movie_idx, "chunk_idx": chunk_idx, "total_chunks": total}
                )

        # Encode all chunks in one call
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_embeddings = np.asarray(self.chunk_embeddings)
        self.chunk_metadata = chunk_metadata

        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        CHUNK_METADATA_PATH.write_text(
            json.dumps({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, indent=2),
            encoding="utf-8",
        )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict[str, Any]]) -> np.ndarray:
        self.documents = documents
        self.document_map = {int(d["id"]): d for d in documents}

        if CHUNK_EMBEDDINGS_PATH.exists() and CHUNK_METADATA_PATH.exists():
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            meta = json.loads(CHUNK_METADATA_PATH.read_text(encoding="utf-8"))
            self.chunk_metadata = meta.get("chunks", [])
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        """Search across chunk embeddings using cosine similarity, aggregate to movies (max score per movie)."""
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise RuntimeError("Chunk embeddings/metadata not loaded. Call load_or_create_chunk_embeddings() first.")

        q = self.generate_embedding(query)
        q = q.astype(float, copy=False)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []

        chunk_vecs = np.asarray(self.chunk_embeddings, dtype=float)
        dots = chunk_vecs @ q
        chunk_norms = np.linalg.norm(chunk_vecs, axis=1)
        denom = chunk_norms * q_norm
        # avoid division by zero
        scores = np.zeros_like(dots, dtype=float)
        mask = denom != 0
        scores[mask] = dots[mask] / denom[mask]

        # build chunk score dicts (as per assignment)
        chunk_scores: list[dict] = []
        for i, sc in enumerate(scores):
            meta = self.chunk_metadata[i]
            chunk_scores.append(
                {"chunk_idx": int(meta["chunk_idx"]), "movie_idx": int(meta["movie_idx"]), "score": float(sc)}
            )

        # aggregate to movie score = best chunk score
        movie_scores: dict[int, float] = {}
        best_chunk: dict[int, int] = {}
        for cs in chunk_scores:
            mi = cs["movie_idx"]
            sc = cs["score"]
            if (mi not in movie_scores) or (sc > movie_scores[mi]):
                movie_scores[mi] = sc
                best_chunk[mi] = cs["chunk_idx"]

        # top movies by score desc
        ranked = sorted(movie_scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]

        results: list[dict] = []
        for movie_idx, sc in ranked:
            doc = self.documents[movie_idx]
            doc_id = doc.get("id")
            title = doc.get("title", "")
            document = (doc.get("description") or "")
            results.append(
                {
                    "id": doc_id,
                    "title": title,
                    "document": document[:100],
                    "score": round(float(sc), SCORE_PRECISION),
                    "metadata": {"movie_idx": int(movie_idx), "chunk_idx": int(best_chunk[movie_idx])},
                }
            )
        return results

