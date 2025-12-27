import json
import os
import string
from pathlib import Path
from typing import Any

from nltk.stem import PorterStemmer

DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
STOPWORDS_PATH = PROJECT_ROOT / "data" / "stopwords.txt"
CACHE_DIR = PROJECT_ROOT / "cache"

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_STEMMER = PorterStemmer()


def load_movies() -> list[dict]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    if not STOPWORDS_PATH.exists():
        return []
    return STOPWORDS_PATH.read_text(encoding="utf-8").splitlines()


def _normalize(text: str) -> str:
    return text.translate(_PUNCT_TABLE).lower().strip()


# normalize stopwords the same way we normalize tokens
_STOPWORDS = {_normalize(w) for w in load_stopwords() if w.strip()}


def tokenize(text: str) -> list[str]:
    # lowercase + remove punctuation
    tokens = [t for t in _normalize(text).split() if t]
    # remove stopwords
    tokens = [t for t in tokens if t not in _STOPWORDS]
    # stemming
    return [_STEMMER.stem(t) for t in tokens]


def format_search_result(
    doc_id: int | str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
