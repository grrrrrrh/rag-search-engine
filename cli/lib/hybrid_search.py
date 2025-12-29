import os
from typing import Any

from lib.keyword_search import InvertedIndex
from lib.search_utils import load_movies


def _dbg(msg: str) -> None:
    if os.getenv("DEBUG_RAG", ""):
        print(f"[DEBUG] {msg}")


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return [1.0 for _ in scores]
    span = mx - mn
    return [(s - mn) / span for s in scores]


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / float(k + rank)


class HybridSearch:
    def __init__(self) -> None:
        self._movies = load_movies()
        self._docmap: dict[int, dict[str, Any]] = {int(m["id"]): m for m in self._movies}
        self._idx: InvertedIndex | None = None

        # Chunked semantic search object is created lazily so that commands that
        # only need BM25 don't pay the embedding load cost.
        self._chunk_search: Any | None = None

    def _get_idx(self) -> InvertedIndex:
        if self._idx is None:
            _dbg("Loading InvertedIndex from cache...")
            self._idx = InvertedIndex.load()
        return self._idx

    def _get_chunk_search(self):
        if self._chunk_search is None:
            _dbg("Loading ChunkedSemanticSearch + chunk embeddings...")
            from lib.semantic_search import ChunkedSemanticSearch

            ss = ChunkedSemanticSearch()
            ss.load_or_create_chunk_embeddings(self._movies)
            self._chunk_search = ss
        return self._chunk_search

    def _bm25_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        idx = self._get_idx()
        return idx.bm25_search(query, limit=limit)

    def _semantic_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        ss = self._get_chunk_search()
        # support either method name depending on your earlier steps
        if hasattr(ss, "search"):
            return ss.search(query, limit=limit)
        return ss.search_chunks(query, limit=limit)

    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = 5) -> list[dict[str, Any]]:
        alpha = max(0.0, min(1.0, float(alpha)))
        big_limit = max(limit * 500, limit)

        bm25 = self._bm25_search(query, big_limit)
        sem = self._semantic_search(query, big_limit)

        bm_map = {int(d["id"]): float(d.get("score", 0.0)) for d in bm25}
        se_map = {int(d["id"]): float(d.get("score", 0.0)) for d in sem}

        cand = sorted(set(bm_map.keys()) | set(se_map.keys()))
        bm_vals = [bm_map.get(i, 0.0) for i in cand]
        se_vals = [se_map.get(i, 0.0) for i in cand]

        bm_norm = dict(zip(cand, normalize_scores(bm_vals)))
        se_norm = dict(zip(cand, normalize_scores(se_vals)))

        combined: list[tuple[int, float]] = []
        for doc_id in cand:
            score = alpha * bm_norm.get(doc_id, 0.0) + (1.0 - alpha) * se_norm.get(doc_id, 0.0)
            combined.append((doc_id, score))

        combined.sort(key=lambda kv: (-kv[1], kv[0]))
        out: list[dict[str, Any]] = []
        for doc_id, score in combined[:limit]:
            doc = self._docmap.get(int(doc_id), {})
            out.append(
                {
                    "id": str(doc_id),
                    "title": doc.get("title", ""),
                    "document": (doc.get("description", "") or ""),
                    "score": float(score),
                }
            )
        return out

    def rrf_search(self, query: str, k: int = 60, limit: int = 5) -> list[dict[str, Any]]:
        k = int(k)
        big_limit = max(limit * 500, limit)

        bm25 = self._bm25_search(query, big_limit)
        sem = self._semantic_search(query, big_limit)

        bm_rank: dict[int, int] = {}
        for i, d in enumerate(bm25, 1):
            bm_rank[int(d["id"])] = i

        se_rank: dict[int, int] = {}
        for i, d in enumerate(sem, 1):
            se_rank[int(d["id"])] = i

        all_ids = set(bm_rank.keys()) | set(se_rank.keys())

        results: list[dict[str, Any]] = []
        for doc_id in all_ids:
            score = 0.0
            br = bm_rank.get(doc_id)
            sr = se_rank.get(doc_id)
            if br is not None:
                score += rrf_score(br, k=k)
            if sr is not None:
                score += rrf_score(sr, k=k)

            doc = self._docmap.get(int(doc_id), {})
            results.append(
                {
                    "id": str(doc_id),
                    "title": doc.get("title", ""),
                    "document": (doc.get("description", "") or ""),
                    "rrf_score": float(score),
                    "bm25_rank": br,
                    "semantic_rank": sr,
                }
            )

        results.sort(key=lambda d: (-d["rrf_score"], int(d["id"])))
        for i, d in enumerate(results, 1):
            d["rrf_rank"] = i
        return results[:limit]
