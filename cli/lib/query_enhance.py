import json
import os
import re
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from google import genai

# You can override this from env if you want:
#   export GEMINI_MODEL="..."
_DEFAULT_MODEL_CANDIDATES = [
    os.getenv("GEMINI_MODEL", "").strip(),
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-002",
]
_DEFAULT_MODEL_CANDIDATES = [m for m in _DEFAULT_MODEL_CANDIDATES if m]


def _debug(msg: str) -> None:
    if os.getenv("DEBUG_RAG") == "1":
        print(f"[DEBUG] {msg}")


@lru_cache(maxsize=1)
def _get_client() -> genai.Client | None:
    load_dotenv()
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        _debug("No GEMINI_API_KEY/GOOGLE_API_KEY found; Gemini calls disabled.")
        return None
    return genai.Client(api_key=key)


def _call_gemini(prompt: str) -> str:
    client = _get_client()
    if client is None:
        return ""

    last_err: Exception | None = None
    for model in _DEFAULT_MODEL_CANDIDATES:
        try:
            _debug(f"Gemini generate_content model={model!r}")
            resp = client.models.generate_content(model=model, contents=prompt)
            text = (getattr(resp, "text", "") or "").strip()
            if text:
                return text
        except Exception as e:
            last_err = e
            _debug(f"Gemini model {model!r} failed: {e}")

    if last_err:
        raise last_err
    return ""


def _clean_single_line(text: str) -> str:
    text = (text or "").strip()
    # remove surrounding quotes
    text = text.strip().strip('"').strip("'").strip()
    # collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


def enhance_query_spell(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return query

    prompt = f"""Fix spelling mistakes in this movie search query.
Return ONLY the corrected query text. No quotes, no explanations.

Query: {query}
"""
    out = _clean_single_line(_call_gemini(prompt))
    return out or query


def enhance_query_rewrite(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return query

    prompt = f"""Rewrite this movie search query to be more optimal for search.
Keep the same intent.
Return ONLY the rewritten query text. No quotes, no explanations.

Query: {query}
"""
    out = _clean_single_line(_call_gemini(prompt))
    return out or query


def enhance_query_expand(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return query

    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Return ONLY the extra terms to append (do NOT repeat the original query).
No quotes, no arrows, no explanations.

Examples:
- scary bear movie -> scary horror grizzly bear terrifying forest attack
- action movie with bear -> action thriller bear chase fight adventure
- comedy with bear -> comedy funny bear humor lighthearted

Query: "{query}"
"""
    extra = _clean_single_line(_call_gemini(prompt))

    if not extra:
        return query

    # If the model still repeats the query, trim it off.
    if extra.lower().startswith(query.lower()):
        extra = _clean_single_line(extra[len(query):])

    if not extra:
        return query

    return f"{query} {extra}".strip()


def evaluate_results_llm(query: str, results: list[dict[str, Any]]) -> list[int]:
    """
    Returns a list of ints 0..3 (same length/order as results).
    """
    query = (query or "").strip()
    if not results:
        return []

    formatted_results: list[str] = []
    for r in results:
        title = str(r.get("title", "")).strip()
        doc = str(r.get("document", "")).strip()
        # keep it short-ish; snippets are enough for judging
        doc = re.sub(r"\s+", " ", doc)[:240]
        formatted_results.append(f"- {title}: {doc}")

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]
"""
    raw = _call_gemini(prompt).strip()
    _debug(f"LLM eval raw: {raw[:200]}")

    # Try strict JSON first
    scores: list[int] | None = None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            scores = [int(x) for x in parsed]
    except Exception:
        scores = None

    # If Gemini wrapped JSON in text, extract the first [...] block
    if scores is None:
        m = re.search(r"\[[\s\S]*\]", raw)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, list):
                    scores = [int(x) for x in parsed]
            except Exception:
                scores = None

    if scores is None:
        scores = [0] * len(results)

    # sanitize + length fix
    clean: list[int] = []
    for x in scores[: len(results)]:
        if x not in (0, 1, 2, 3):
            x = 0
        clean.append(x)
    while len(clean) < len(results):
        clean.append(0)

    return clean
