import math
import os
import pickle
from collections import Counter
from typing import Callable

from search_utils import CACHE_DIR, BM25_B, BM25_K1, tokenize


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str, tokenize_fn: Callable[[str], list[str]] = tokenize) -> None:
        tokens = tokenize_fn(text)

        # BM25 length is token count INCLUDING duplicates
        self.doc_lengths[doc_id] = len(tokens)

        self.term_frequencies.setdefault(doc_id, Counter())
        for tok in tokens:
            self.index.setdefault(tok, set()).add(doc_id)
            self.term_frequencies[doc_id][tok] += 1

    def build(self, movies: list[dict], tokenize_fn: Callable[[str], list[str]] = tokenize) -> None:
        for m in movies:
            doc_id = int(m["id"])
            self.docmap[doc_id] = m
            text = f"{m.get('title','')} {m.get('description','')}"
            self.__add_document(doc_id, text, tokenize_fn)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    @classmethod
    def load(cls):
        obj = cls()
        required = [obj.index_path, obj.docmap_path, obj.term_frequencies_path, obj.doc_lengths_path]
        if not all(os.path.exists(p) for p in required):
            raise FileNotFoundError("Missing cache files. Run `build` first.")

        with open(obj.index_path, "rb") as f:
            obj.index = pickle.load(f)
        with open(obj.docmap_path, "rb") as f:
            obj.docmap = pickle.load(f)
        with open(obj.term_frequencies_path, "rb") as f:
            obj.term_frequencies = pickle.load(f)
        with open(obj.doc_lengths_path, "rb") as f:
            obj.doc_lengths = pickle.load(f)

        return obj

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term.lower(), set()))

    # ---------- token helpers (NO extra tokenization) ----------

    def _tf_token(self, doc_id: int, tok: str) -> int:
        c = self.term_frequencies.get(int(doc_id))
        if c is None:
            return 0
        return int(c.get(tok, 0))

    def _avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def _bm25_idf_token(self, tok: str) -> float:
        N = len(self.docmap)
        df = len(self.index.get(tok, set()))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def _bm25_tf_token(self, doc_id: int, tok: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self._tf_token(doc_id, tok)
        if tf == 0:
            return 0.0

        dl = float(self.doc_lengths.get(int(doc_id), 0))
        avgdl = self._avg_doc_length()

        length_norm = 1.0
        if avgdl > 0.0:
            length_norm = 1 - b + b * (dl / avgdl)

        denom = tf + k1 * length_norm
        return (tf * (k1 + 1)) / denom if denom != 0 else 0.0

    # ---------- public API (tokenizes & validates single token) ----------

    def __single_token(self, term: str) -> str:
        toks = tokenize(term)
        if len(toks) != 1:
            raise ValueError("Term must tokenize to exactly one token.")
        return toks[0]

    def get_tf(self, doc_id: int, term: str) -> int:
        tok = self.__single_token(term)
        return self._tf_token(doc_id, tok)

    def get_bm25_idf(self, term: str) -> float:
        tok = self.__single_token(term)
        return self._bm25_idf_token(tok)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tok = self.__single_token(term)
        return self._bm25_tf_token(doc_id, tok, k1=k1, b=b)
        q_tokens = tokenize(query)

        # Precompute IDF per query token once (tokens already normalized/stemmed)
        q_idf = {tok: self._bm25_idf_token(tok) for tok in q_tokens}

        scores: dict[int, float] = {}
        for doc_id in sorted(self.docmap.keys()):
            total = 0.0
            for tok in q_tokens:
                total += self._bm25_tf_token(doc_id, tok) * q_idf[tok]
            if total > 0.0:
                scores[doc_id] = total

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:limit]

    def bm25(self, doc_id: int, term: str) -> float:
        # full BM25 for a single term in a single document
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int = 5) -> list[tuple[int, float]]:
        # assignment version: score EVERY document by summing BM25 over query tokens
        q_tokens = tokenize(query)

        scores: dict[int, float] = {}
        for doc_id in self.docmap.keys():
            total = 0.0
            for tok in q_tokens:
                total += self.bm25(doc_id, tok)
            scores[doc_id] = total

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:limit]
