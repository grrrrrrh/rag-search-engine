import string
from pathlib import Path
from nltk.stem import PorterStemmer

PUNCT_TABLE = str.maketrans("", "", string.punctuation)
stemmer = PorterStemmer()


def normalize(text: str) -> str:
    return text.translate(PUNCT_TABLE).lower().strip()


def load_stopwords() -> set[str]:
    path = Path("data") / "stopwords.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    # normalize so "aren't" becomes "arent", matching punctuation-stripped tokens
    return {normalize(line) for line in lines if line.strip()}


STOPWORDS = load_stopwords()


def tokenize(text: str) -> list[str]:
    tokens = [tok for tok in normalize(text).split() if tok]
    tokens = [tok for tok in tokens if tok not in STOPWORDS]
    return [stemmer.stem(tok) for tok in tokens]
