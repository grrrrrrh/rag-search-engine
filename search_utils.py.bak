BM25_B = 0.75
BM25_K1 = 1.5
CACHE_DIR = "cache"

import string
from pathlib import Path
from nltk.stem import PorterStemmer

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_STEMMER = PorterStemmer()


def _normalize(text: str) -> str:
    return text.translate(_PUNCT_TABLE).lower().strip()


def _load_stopwords() -> set[str]:
    path = Path("data") / "stopwords.txt"
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    # normalize so "aren't" becomes "arent", matching punctuation-stripped tokens
    return {_normalize(line) for line in lines if line.strip()}


_STOPWORDS = _load_stopwords()


def tokenize(text: str) -> list[str]:
    tokens = [tok for tok in _normalize(text).split() if tok]
    tokens = [tok for tok in tokens if tok not in _STOPWORDS]
    return [_STEMMER.stem(tok) for tok in tokens]
