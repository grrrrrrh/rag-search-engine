#!/usr/bin/env python3

import argparse
import mimetypes
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Describe an image and rewrite a query for better movie search")
    parser.add_argument("--image", required=True, type=str, help="Path to an image file")
    parser.add_argument("--query", required=True, type=str, help="Text query to rewrite based on the image")
    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        img = f.read()

    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GENAI_API_KEY")
    )
    if not api_key:
        raise SystemExit(
            "Missing API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY / GENAI_API_KEY) in your environment or .env"
        )

    client = genai.Client(api_key=api_key)

    system_prompt = (
        "Given the included image and text query, rewrite the text query to improve search results "
        "from a movie database. Make sure to:\n"
        "- Synthesize visual and textual information\n"
        "- Focus on movie-specific details (actors, scenes, style, etc.)\n"
        "- Return only the rewritten query, without any additional commentary"
    )

    parts = [
        system_prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]

    # Keep this deterministic-ish
    resp = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        contents=parts,
        config=types.GenerateContentConfig(temperature=0),
    )

    print(f"Rewritten query: {resp.text.strip()}")
    if resp.usage_metadata is not None:
        print(f"Total tokens:    {resp.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
