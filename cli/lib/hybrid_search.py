from __future__ import annotations

from typing import Any

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


def normalize(scores: list[float]) -> list[float]:
    """Min-max normalize to [0,1]. If all same -> all 1.0."""
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    if mn == mx:
        return [1.0] * len(scores)
    denom = mx - mn
    return [(s - mn) / denom for s in scores]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1.0 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    # rank is 1-based
    if rank <= 0:
        return 0.0
    return 1.0 / (k + rank)


def _coerce_pairs(raw: Any) -> list[tuple[int, float]]:
    """
    Coerce bm25 output into [(doc_id, score), ...] for common shapes:
      - [(doc_id, score), ...]
      - [(doc_id, title, score, ...), ...]
      - [{"id": doc_id, "score": score, ...}, ...]
    """
    pairs: list[tuple[int, float]] = []
    if not raw:
        return pairs

    for item in raw:
        doc_id = None
        score = 0.0

        if isinstance(item, dict):
            doc_id = item.get("id")
            score = item.get("score", 0.0)
        else:
            try:
                doc_id = item[0]
                score = item[1]
                # sometimes item[1] is title, and item[2] is score
                if isinstance(score, str) and len(item) >= 3:
                    score = item[2]
            except Exception:
                continue

        try:
            pairs.append((int(doc_id), float(score)))
        except Exception:
            continue

    return pairs


class HybridSearch:
    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = documents
        self.doc_by_id: dict[int, dict[str, Any]] = {}
        for d in documents:
            try:
                self.doc_by_id[int(d["id"])] = d
            except Exception:
                continue

        # Semantic chunk search
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        # BM25 index
        self.idx = InvertedIndex()
        try:
            self.idx.load()
        except Exception:
            # build/save once if missing
            self.idx.build()
            self.idx.save()
            self.idx.load()

    def _bm25_search(self, query: str, limit: int):
        # ensure loaded
        try:
            self.idx.load()
        except Exception:
            pass
        return self.idx.bm25_search(query, limit)

    def _semantic_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        # course sometimes calls this "search", sometimes "search_chunks"
        fn = getattr(self.semantic_search, "search", None) or getattr(self.semantic_search, "search_chunks", None)
        if fn is None:
            raise RuntimeError("ChunkedSemanticSearch has no search/search_chunks method")

        try:
            return fn(query, limit=limit)
        except TypeError:
            return fn(query, limit)

    def _snippet(self, doc: dict[str, Any], max_len: int = 100) -> str:
        desc = (doc.get("description") or "").strip()
        desc = " ".join(desc.split())
        if not desc:
            return "..."
        if len(desc) <= max_len:
            return desc
        return desc[:max_len].rstrip() + "..."

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict[str, Any]]:
        big_limit = limit * 500

        bm25_pairs = _coerce_pairs(self._bm25_search(query, big_limit))
        bm25_ids = [d for d, _ in bm25_pairs]
        bm25_scores = [s for _, s in bm25_pairs]
        bm25_norm = normalize(bm25_scores)
        bm25_by_id = {doc_id: bm25_norm[i] for i, doc_id in enumerate(bm25_ids)}

        sem_raw = self._semantic_search(query, big_limit)
        sem_ids: list[int] = []
        sem_scores: list[float] = []
        for r in sem_raw:
            try:
                sem_ids.append(int(r["id"]))
                sem_scores.append(float(r["score"]))
            except Exception:
                continue
        sem_norm = normalize(sem_scores)
        sem_by_id = {doc_id: sem_norm[i] for i, doc_id in enumerate(sem_ids)}

        all_ids = set(bm25_by_id.keys()) | set(sem_by_id.keys())

        merged: list[dict[str, Any]] = []
        for doc_id in all_ids:
            doc = self.doc_by_id.get(doc_id)
            if not doc:
                continue
            bm25_s = bm25_by_id.get(doc_id, 0.0)
            sem_s = sem_by_id.get(doc_id, 0.0)
            merged.append(
                {
                    "id": doc_id,
                    "title": doc.get("title", ""),
                    "document": self._snippet(doc),
                    "bm25": bm25_s,
                    "semantic": sem_s,
                    "hybrid": hybrid_score(bm25_s, sem_s, alpha=alpha),
                }
            )

        merged.sort(key=lambda r: (-r["hybrid"], r["id"]))
        return merged[:limit]

    def rrf_search(self, query: str, k: int = 60, limit: int = 5) -> list[dict[str, Any]]:
        big_limit = limit * 500

        bm25_pairs = _coerce_pairs(self._bm25_search(query, big_limit))
        sem_raw = self._semantic_search(query, big_limit)

        combined: dict[int, dict[str, Any]] = {}

        # BM25 ranks (1-based)
        for rank, (doc_id, _score) in enumerate(bm25_pairs, start=1):
            doc = self.doc_by_id.get(doc_id)
            if not doc:
                continue
            entry = combined.get(doc_id)
            if entry is None:
                entry = {
                    "id": doc_id,
                    "title": doc.get("title", ""),
                    "document": self._snippet(doc),
                    "rrf": 0.0,
                    "bm25_rank": None,
                    "semantic_rank": None,
                }
                combined[doc_id] = entry

            entry["bm25_rank"] = rank
            entry["rrf"] += rrf_score(rank, k)

        # Semantic ranks (dedupe per movie id; first occurrence is best rank)
        seen_sem: set[int] = set()
        for rank, r in enumerate(sem_raw, start=1):
            try:
                doc_id = int(r["id"])
            except Exception:
                continue
            if doc_id in seen_sem:
                continue
            seen_sem.add(doc_id)

            doc = self.doc_by_id.get(doc_id)
            if not doc:
                continue

            entry = combined.get(doc_id)
            if entry is None:
                entry = {
                    "id": doc_id,
                    "title": doc.get("title", ""),
                    "document": self._snippet(doc),
                    "rrf": 0.0,
                    "bm25_rank": None,
                    "semantic_rank": None,
                }
                combined[doc_id] = entry

            entry["semantic_rank"] = rank
            entry["rrf"] += rrf_score(rank, k)

        results = list(combined.values())
        results.sort(key=lambda e: (-float(e["rrf"]), int(e["id"])))
        return results[:limit]
