from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from google import genai


def _get_api_key() -> str:
    load_dotenv()
    key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENAI_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "Missing API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or .env file."
        )
    return key


def _pick_models() -> list[str]:
    # Prefer env override, otherwise try a few common working models.
    env_model = os.getenv("GEMINI_MODEL")
    candidates = [env_model] if env_model else []
    candidates += [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    # remove Nones/empties + de-dupe while preserving order
    out: list[str] = []
    for m in candidates:
        if m and m not in out:
            out.append(m)
    return out


def _call_gemini(prompt: str) -> str:
    client = genai.Client(api_key=_get_api_key())

    last_err: Exception | None = None
    for model in _pick_models():
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            text = (getattr(resp, "text", None) or "").strip()
            if text:
                return text
            # Fallback if .text isn't populated for some reason
            return str(resp).strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue

    raise RuntimeError(f"All Gemini model attempts failed. Last error: {last_err!r}")


def _normalize_result(r: Any) -> dict[str, str]:
    # HybridSearch results are usually dicts with title/document, but we make it robust.
    if isinstance(r, dict):
        title = (
            r.get("title")
            or r.get("movie_title")
            or r.get("name")
            or r.get("id")
            or "Untitled"
        )
        doc = r.get("document") or r.get("description") or r.get("text") or ""
        return {"title": str(title), "document": str(doc)}
    return {"title": str(r), "document": ""}


def retrieve_results(query: str, limit: int = 5) -> list[dict[str, str]]:
    # Uses your hybrid pipeline (RRF). That’s what your course has been building.
    from lib.hybrid_search import HybridSearch  # local import to avoid import-time cost

    hs = HybridSearch()
    results = hs.rrf_search(query, limit=limit)
    return [_normalize_result(r) for r in results]


def summarize_query(query: str, limit: int = 5) -> None:
    query = (query or "").strip()
    results = retrieve_results(query, limit=limit)

    sources_block = "\n".join(
        f"- {i+1}. {r['title']}: {r['document']}" for i, r in enumerate(results)
    )

    prompt = f"""You are helping a user search a movie dataset.

User query: "{query}"

Relevant movies (title + description snippet):
{sources_block}

Write a short helpful answer (1 paragraph). Do NOT include citations.
"""
    print(_call_gemini(prompt))


def answer_with_citations(query: str, limit: int = 5) -> None:
    query = (query or "").strip()
    results = retrieve_results(query, limit=limit)

    print(f"Search Results for query '{query}':")
    for r in results:
        print(f"- {r['title']}")

    sources_block = "\n\n".join(
        f"Source [{i+1}] — {r['title']}:\n{r['document']}"
        for i, r in enumerate(results)
    )

    prompt = f"""Answer the user's movie search query using ONLY the sources below.

Rules:
- If you use information from a source, cite it like [1] or [2] at the end of the sentence.
- You may cite multiple sources like [1][3].
- If the sources don't support a claim, say you don't know.

User query: "{query}"

Sources:
{sources_block}
"""

    print("\nLLM Answer:")
    print(_call_gemini(prompt))


def answer_question(question: str, limit: int = 5) -> None:
    """
    Answer a natural-language question using retrieved movie descriptions as context.
    """
    question = (question or "").strip()
    results = retrieve_results(question, limit=limit)

    sources_block = "\n\n".join(
        f"Source [{i+1}] — {r['title']}:\n{r['document']}"
        for i, r in enumerate(results)
    )

    prompt = f"""You are answering a user's question about movies using ONLY the sources below.

Rules:
- Use ONLY the sources.
- If the answer isn't supported by the sources, say you don't know.
- Keep the answer short and direct.

Question: "{question}"

Sources:
{sources_block}

Answer:
"""
    print(_call_gemini(prompt))
