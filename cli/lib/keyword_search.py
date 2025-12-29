import json
import math
import os
import pickle
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from nltk.stem import PorterStemmer

from lib.search_utils import (
    BM25_B,
    BM25_K1,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    format_search_result,
    load_movies,
    load_stopwords,
)

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_STEMMER = PorterStemmer()

_STOPWORDS: set[str] | None = None


def _normalize(text: str) -> str:
    return text.translate(_PUNCT_TABLE).lower().strip()


def _get_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        try:
            _STOPWORDS = {_normalize(w) for w in load_stopwords() if w.strip()}
        except FileNotFoundError:
            _STOPWORDS = set()
    return _STOPWORDS


def tokenize_text(text: str) -> list[str]:
    text = _normalize(text or "")
    if not text:
        return []
    toks = [t for t in text.split() if t]
    sw = _get_stopwords()
    toks = [t for t in toks if t not in sw]
    return [_STEMMER.stem(t) for t in toks]


class InvertedIndex:
    """
    Simple inverted index storing postings as:
        postings[token][doc_id] = tf
    """

    CACHE_PATH = os.path.join(CACHE_DIR, "inverted_index.pkl")

    def __init__(self) -> None:
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.doc_lengths: dict[int, int] = {}
        self.documents: list[dict[str, Any]] = []
        self.document_map: dict[int, dict[str, Any]] = {}

    def build(self) -> None:
        self.postings = defaultdict(dict)
        self.doc_lengths = {}
        self.documents = load_movies()

        # map doc_id -> full document
        self.document_map = {}
        for doc in self.documents:
            doc_id = int(doc.get("id"))
            self.document_map[doc_id] = doc

        for doc_id, doc in self.document_map.items():
            title = doc.get("title", "") or ""
            desc = doc.get("description", "") or ""
            text = f"{title} {desc}"
            tokens = tokenize_text(text)
            self.doc_lengths[doc_id] = len(tokens)

            counts = Counter(tokens)
            for tok, tf in counts.items():
                self.postings[tok][doc_id] = tf

        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.CACHE_PATH, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls) -> "InvertedIndex":
        """
        Load from cache if possible; otherwise rebuild.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        if os.path.exists(cls.CACHE_PATH):
            with open(cls.CACHE_PATH, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, cls):
                return obj

        idx = cls()
        idx.build()
        return idx

    # ---------- helpers ----------

    def _avgdl(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / float(len(self.doc_lengths))

    def _single_token(self, term: str) -> str:
        toks = tokenize_text(term)
        if len(toks) != 1:
            raise ValueError("Term must tokenize to exactly one token.")
        return toks[0]

    def _tf_token(self, doc_id: int, tok: str) -> int:
        return int(self.postings.get(tok, {}).get(int(doc_id), 0))

    # ---------- public API ----------

    def get_tf(self, doc_id: int, term: str) -> int:
        tok = self._single_token(term)
        return self._tf_token(doc_id, tok)

    def get_idf(self, term: str) -> float:
        # classic IDF for TF-IDF style (kept for earlier commands)
        tok = self._single_token(term)
        N = max(len(self.document_map), 1)
        df = len(self.postings.get(tok, {}))
        if df == 0:
            return 0.0
        return math.log(N / df)

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        return float(tf) * self.get_idf(term)

    def get_bm25_idf(self, term: str) -> float:
        tok = self._single_token(term)
        N = len(self.document_map)
        df = len(self.postings.get(tok, {}))
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tok = self._single_token(term)
        tf = self._tf_token(doc_id, tok)
        if tf == 0:
            return 0.0

        dl = float(self.doc_lengths.get(int(doc_id), 0))
        avgdl = self._avgdl()

        length_norm = 1.0
        if avgdl > 0.0:
            length_norm = 1.0 - b + b * (dl / avgdl)

        denom = tf + k1 * length_norm
        return (tf * (k1 + 1.0)) / denom if denom != 0 else 0.0

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict[str, Any]]:
        """
        Earlier "BM25" search (TF part only): sum BM25_TF over query tokens.
        """
        q_tokens = tokenize_text(query)
        scores: dict[int, float] = {}

        for doc_id in self.document_map.keys():
            total = 0.0
            for tok in q_tokens:
                # token already normalized/stemmed; use token-API directly
                tf = self._tf_token(doc_id, tok)
                if tf == 0:
                    continue
                # reuse bm25_tf math but with token directly
                dl = float(self.doc_lengths.get(int(doc_id), 0))
                avgdl = self._avgdl()
                length_norm = 1.0 - BM25_B + BM25_B * (dl / avgdl) if avgdl > 0 else 1.0
                denom = tf + BM25_K1 * length_norm
                total += (tf * (BM25_K1 + 1.0)) / denom if denom != 0 else 0.0

            if total > 0:
                scores[int(doc_id)] = total

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        out: list[dict[str, Any]] = []
        for doc_id, score in ranked:
            doc = self.document_map[int(doc_id)]
            out.append(
                format_search_result(
                    str(doc_id),
                    doc.get("title", ""),
                    doc.get("description", "") or "",
                    score,
                )
            )
        return out

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict[str, Any]]:
        """
        Full BM25: sum BM25(doc, token) over query tokens.
        """
        q_tokens = tokenize_text(query)
        if not q_tokens:
            return []

        # precompute token idf once
        N = len(self.document_map)
        q_idf: dict[str, float] = {}
        for tok in q_tokens:
            df = len(self.postings.get(tok, {}))
            q_idf[tok] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        scores: dict[int, float] = {}
        avgdl = self._avgdl()

        for doc_id in self.document_map.keys():
            dl = float(self.doc_lengths.get(int(doc_id), 0))
            length_norm = 1.0 - BM25_B + BM25_B * (dl / avgdl) if avgdl > 0 else 1.0

            total = 0.0
            for tok in q_tokens:
                tf = self._tf_token(doc_id, tok)
                if tf == 0:
                    continue
                denom = tf + BM25_K1 * length_norm
                bm25_tf = (tf * (BM25_K1 + 1.0)) / denom if denom != 0 else 0.0
                total += bm25_tf * q_idf[tok]

            if total > 0.0:
                scores[int(doc_id)] = total

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        out: list[dict[str, Any]] = []
        for doc_id, score in ranked:
            doc = self.document_map[int(doc_id)]
            out.append(
                format_search_result(
                    str(doc_id),
                    doc.get("title", ""),
                    doc.get("description", "") or "",
                    score,
                )
            )
        return out


# ---------- command functions used by the CLI ----------

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict[str, Any]]:
    idx = InvertedIndex.load()
    return idx.search(query, limit=limit)


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
    return idx.bm25_search(query, limit=limit)
