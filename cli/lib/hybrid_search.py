from __future__ import annotations

from typing import Any

from lib.search_utils import load_movies, format_search_result

# Be resilient to where InvertedIndex lives in your project
try:
    from lib.keyword_search import InvertedIndex
except Exception:  # pragma: no cover
    from inverted_index import InvertedIndex  # type: ignore

try:
    from lib.semantic_search import ChunkedSemanticSearch
except Exception:  # pragma: no cover
    from semantic_search import ChunkedSemanticSearch  # type: ignore


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize scores to [0,1]. If all equal, return all 1.0."""
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        return [1.0 for _ in scores]
    rng = hi - lo
    return [(s - lo) / rng for s in scores]


def rrf_score(rank: int, k: int = 60) -> float:
    # rank is 1-based
    return 1.0 / (k + rank)


class HybridSearch:
    def __init__(self, documents: list[dict] | None = None) -> None:
        self.documents = documents or load_movies()
        self.doc_by_id: dict[int, dict] = {int(d["id"]): d for d in self.documents}

        # Load chunk embeddings (cached if present)
        self.semantic = ChunkedSemanticSearch()
        self.semantic.load_or_create_chunk_embeddings(self.documents)

        # Load BM25 index (must exist from earlier build step)
        # Load BM25 index whether load() is a classmethod OR an instance method
        try:
            self.idx = InvertedIndex.load()
        except TypeError:
            idx = InvertedIndex()
            rv = idx.load()
            self.idx = rv if isinstance(rv, InvertedIndex) else idx


    def _bm25_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        raw = self.idx.bm25_search(query, limit=limit)

        out: list[dict[str, Any]] = []
        for item in raw:
            # support either [(doc_id, score), ...] or [{"id":..,"score":..}, ...]
            if isinstance(item, dict):
                doc_id = int(item["id"])
                score = float(item["score"])
            else:
                doc_id = int(item[0])
                score = float(item[1])

            doc = self.doc_by_id.get(doc_id)
            if not doc:
                continue

            out.append(
                format_search_result(
                    str(doc_id),
                    doc.get("title", ""),
                    doc.get("description", "") or "",
                    score,
                )
            )
        return out

    def _semantic_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        # Some projects call it search_chunks, some call it search; support both.
        if hasattr(self.semantic, "search_chunks"):
            return self.semantic.search_chunks(query, limit=limit)  # type: ignore[attr-defined]
        return self.semantic.search(query, limit=limit)  # type: ignore[attr-defined]

    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = 5) -> list[dict[str, Any]]:
        big_limit = max(1, limit * 500)

        bm25 = self._bm25_search(query, big_limit)
        sem = self._semantic_search(query, big_limit)

        bm25_scores = [float(r["score"]) for r in bm25]
        sem_scores = [float(r["score"]) for r in sem]

        bm25_norm = normalize_scores(bm25_scores)
        sem_norm = normalize_scores(sem_scores)

        bm25_by_id = {int(r["id"]): (r, bm25_norm[i]) for i, r in enumerate(bm25)}
        sem_by_id = {int(r["id"]): (r, sem_norm[i]) for i, r in enumerate(sem)}

        all_ids = set(bm25_by_id.keys()) | set(sem_by_id.keys())

        combined: list[dict[str, Any]] = []
        for doc_id in all_ids:
            bm25_r, b = bm25_by_id.get(doc_id, (None, 0.0))
            sem_r, s = sem_by_id.get(doc_id, (None, 0.0))

            base = sem_r or bm25_r
            if base is None:
                continue

            combined_score = alpha * b + (1.0 - alpha) * s
            combined.append(
                {
                    "id": str(doc_id),
                    "title": base["title"],
                    "document": base.get("document", ""),
                    "score": combined_score,
                }
            )

        combined.sort(key=lambda r: (-float(r["score"]), int(r["id"])))
        return combined[:limit]

    def rrf_search(self, query: str, k: int = 60, limit: int = 5) -> list[dict[str, Any]]:
        big_limit = max(1, limit * 500)

        bm25 = self._bm25_search(query, big_limit)
        sem = self._semantic_search(query, big_limit)

        merged: dict[int, dict[str, Any]] = {}

        # BM25 ranks
        for i, r in enumerate(bm25, start=1):
            doc_id = int(r["id"])
            if doc_id not in merged:
                merged[doc_id] = {
                    "id": str(doc_id),
                    "title": r["title"],
                    "document": r.get("document", ""),
                    "bm25_rank": i,
                    "semantic_rank": None,
                    "rrf_score": 0.0,
                }
            merged[doc_id]["bm25_rank"] = i
            merged[doc_id]["rrf_score"] += rrf_score(i, k=k)

        # Semantic ranks
        for i, r in enumerate(sem, start=1):
            doc_id = int(r["id"])
            if doc_id not in merged:
                merged[doc_id] = {
                    "id": str(doc_id),
                    "title": r["title"],
                    "document": r.get("document", ""),
                    "bm25_rank": None,
                    "semantic_rank": i,
                    "rrf_score": 0.0,
                }
            merged[doc_id]["semantic_rank"] = i
            merged[doc_id]["rrf_score"] += rrf_score(i, k=k)

        results = list(merged.values())
        results.sort(key=lambda r: (-float(r["rrf_score"]), int(r["id"])))
        return results[:limit]
