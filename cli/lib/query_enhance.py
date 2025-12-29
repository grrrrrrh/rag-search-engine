from __future__ import annotations

import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions

load_dotenv()

# Allow override (recommended), but provide a sane default.
# gemini-1.5-flash is often unavailable now; docs show newer model names.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_API_VERSION = os.getenv("GEMINI_API_VERSION", "v1beta")


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment/.env"
        )

    # Force a known API version; v1beta is fine for Gemini API, v1 is also supported in some contexts.
    return genai.Client(
        api_key=api_key,
        http_options=HttpOptions(api_version=DEFAULT_API_VERSION),
    )


def _one_line(text: str) -> str:
    # Normalize whitespace and strip quotes/backticks that models sometimes add
    s = (text or "").strip()
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip("`").strip()
    # Strip surrounding quotes if the model returns them
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s


def enhance_query_spell(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return ""

    prompt = f"""Fix spelling mistakes in this movie search query.
Return ONLY the corrected query, nothing else.

Query: "{q}"
"""
    client = _get_client()
    resp = client.models.generate_content(model=DEFAULT_MODEL, contents=prompt)
    return _one_line(getattr(resp, "text", "") or q) or q


def enhance_query_rewrite(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return ""

    prompt = f"""Rewrite this movie search query to be more optimal for retrieval.
Keep the meaning, but make it clearer and more search-friendly.
Return ONLY the rewritten query, nothing else.

Query: "{q}"
"""
    client = _get_client()
    resp = client.models.generate_content(model=DEFAULT_MODEL, contents=prompt)
    return _one_line(getattr(resp, "text", "") or q) or q


def enhance_query_expand(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return ""

    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Return ONLY the expanded terms (not the original query), as a single line.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{q}"
"""
    client = _get_client()
    resp = client.models.generate_content(model=DEFAULT_MODEL, contents=prompt)
    expansion = _one_line(getattr(resp, "text", ""))

    # Append expansion to original query (per assignment)
    if expansion:
        return f"{q} {expansion}".strip()
    return q
