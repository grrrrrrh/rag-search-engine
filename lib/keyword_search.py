import math
import pickle
from collections import Counter
from typing import Any

from lib.search_utils import (
    BM25_B,
    BM25_K1,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    format_search_result,
    load_movies,
    tokenize,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict[str, Any]] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.doc_lengths: dict[int, int] = {}

        self._index_path = CACHE_DIR / "index.pkl"
        self._docmap_path = CACHE_DIR / "docmap.pkl"
        self._tf_path = CACHE_DIR / "term_frequencies.pkl"
        self._dl_path = CACHE_DIR / "doc_lengths.pkl"

    def __add_document(self, doc_id: int, title: str, description: str) -> None:
        title_tokens = tokenize(title or "")
        desc_tokens = tokenize(description or "")
        tokens = title_tokens + desc_tokens

        # BM25 doc length MUST be total token count (incl duplicates) of the processed tokens.
        self.doc_lengths[doc_id] = len(tokens)

        # term frequencies per doc
        self.term_frequencies[doc_id] = Counter(tokens)

        # document frequency index (doc appears once per term)
        for tok in set(tokens):
            self.index.setdefault(tok, set()).add(doc_id)

    def build(self, movies: list[dict]) -> None:
        for m in movies:
            doc_id = int(m["id"])
            self.docmap[doc_id] = m
            self.__add_document(doc_id, m.get("title", ""), m.get("description", ""))

    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with self._index_path.open("wb") as f:
            pickle.dump(self.index, f)
        with self._docmap_path.open("wb") as f:
            pickle.dump(self.docmap, f)
        with self._tf_path.open("wb") as f:
            pickle.dump(self.term_frequencies, f)
        with self._dl_path.open("wb") as f:
            pickle.dump(self.doc_lengths, f)

    @classmethod
    def load(cls) -> "InvertedIndex":
        obj = cls()
        required = [obj._index_path, obj._docmap_path, obj._tf_path, obj._dl_path]
        if not all(p.exists() for p in required):
            raise FileNotFoundError("Missing cache files. Run `build` first.")
        with obj._index_path.open("rb") as f:
            obj.index = pickle.load(f)
        with obj._docmap_path.open("rb") as f:
            obj.docmap = pickle.load(f)
        with obj._tf_path.open("rb") as f:
            obj.term_frequencies = pickle.load(f)
        with obj._dl_path.open("rb") as f:
            obj.doc_lengths = pickle.load(f)
        return obj

    def _avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def __single_token(self, term: str) -> str:
        toks = tokenize(term)
        if len(toks) != 1:
            raise ValueError("Term must tokenize to exactly one token.")
        return toks[0]

    # ---------- TF / IDF ----------
    def get_tf(self, doc_id: int, term: str) -> int:
        tok = self.__single_token(term)
        return int(self.term_frequencies.get(int(doc_id), Counter()).get(tok, 0))

    def get_idf(self, term: str) -> float:
        tok = self.__single_token(term)
        N = len(self.docmap)
        df = len(self.index.get(tok, set()))
        return math.log((N + 1) / (df + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        return float(self.get_tf(doc_id, term)) * self.get_idf(term)

    # ---------- BM25 ----------
    def get_bm25_idf(self, term: str) -> float:
        tok = self.__single_token(term)
        N = len(self.docmap)
        df = len(self.index.get(tok, set()))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tok = self.__single_token(term)
        tf = int(self.term_frequencies.get(int(doc_id), Counter()).get(tok, 0))
        if tf == 0:
            return 0.0

        dl = float(self.doc_lengths.get(int(doc_id), 0))
        avgdl = self._avg_doc_length()
        norm = 1.0 if avgdl <= 0 else (1 - b + b * (dl / avgdl))

        denom = tf + k1 * norm
        return (tf * (k1 + 1)) / denom if denom != 0 else 0.0

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[tuple[int, float]]:
        q_tokens = tokenize(query)

        scores: dict[int, float] = {}
        for doc_id in self.docmap.keys():
            total = 0.0
            for tok in q_tokens:
                # tok is already a single token; bm25() will validate/tokenize to one token again
                total += self.bm25(doc_id, tok)
            scores[doc_id] = total

        # deterministic tie-break: score desc, doc_id asc
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:limit]


# -----------------------
# Command helpers (boot.dev tests call these)
# -----------------------
def build_command() -> None:
    movies = load_movies()
    idx = InvertedIndex()
    idx.build(movies)
    idx.save()
    # boot.dev expects this number in stdout
    print(len(idx.index))


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict[str, Any]]:
    try:
        idx = InvertedIndex.load()
    except FileNotFoundError:
        return []

    results: list[dict[str, Any]] = []
    seen: set[int] = set()

    for tok in tokenize(query):
        for doc_id in sorted(idx.index.get(tok, set())):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            m = idx.docmap[doc_id]
            results.append(format_search_result(doc_id, m.get("title", ""), m.get("description", ""), 0.0))
            if len(results) >= limit:
                return results

    return results


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex.load()
    return idx.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    idx = InvertedIndex.load()
    return idx.get_idf(term)


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex.load()
    return idx.get_tfidf(doc_id, term)


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex.load()
    return idx.get_bm25_idf(term)


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex.load()
    return idx.get_bm25_tf(doc_id, term, k1=k1, b=b)


def bm25search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict[str, Any]]:
    idx = InvertedIndex.load()
    ranked = idx.bm25_search(query, limit=limit)

    out: list[dict[str, Any]] = []
    for doc_id, score in ranked:
        m = idx.docmap[doc_id]
        out.append(
            format_search_result(
                doc_id,
                m.get("title", ""),
                m.get("description", ""),
                score,
            )
        )
    return out
