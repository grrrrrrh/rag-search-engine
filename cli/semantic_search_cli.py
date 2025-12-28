#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    ChunkedSemanticSearch,
    embed_query_text,
    embed_text,
    load_movies,
    semantic_chunk_text,
    verify_embeddings,
    verify_model,
)


def chunk_words(text: str, max_words: int = 10, overlap: int = 0) -> list[str]:
    words = [w for w in (text or "").split() if w]
    if not words:
        return []
    if max_words <= 0:
        raise ValueError("--max-words must be > 0")
    if overlap < 0:
        raise ValueError("--overlap must be >= 0")
    if overlap >= max_words:
        raise ValueError("--overlap must be smaller than --max-words")

    step = max_words - overlap
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + max_words]))
        i += step
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify embedding model loads")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed a text and print info")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify doc embeddings cache / shape")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed a query and print info")
    embed_query_parser.add_argument("query", type=str, help="Query text")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text by words")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--max-words", type=int, default=10, help="Max words per chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Word overlap")

    sem_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk text by sentences")
    sem_chunk_parser.add_argument("text", type=str, help="Text to semantically chunk")
    sem_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Max sentences per chunk")
    sem_chunk_parser.add_argument("--overlap", type=int, default=0, help="Sentence overlap")

    subparsers.add_parser("embed_chunks", help="Build/load chunk embeddings")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search movies using chunk embeddings")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case "chunk":
            text = args.text
            chunks = chunk_words(text, max_words=args.max_words, overlap=args.overlap)
            print(f"Chunking {len(text)} characters")
            for i, ch in enumerate(chunks, 1):
                print(f"{i}. {ch}")

        case "semantic_chunk":
            text = args.text
            chunks = semantic_chunk_text(text, max_chunk_size=args.max_chunk_size, overlap=args.overlap)
            print(f"Semantically chunking {len(text)} characters")
            for i, ch in enumerate(chunks, 1):
                print(f"{i}. {ch}")

        case "embed_chunks":
            movies = load_movies()
            ss = ChunkedSemanticSearch()
            embeddings = ss.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            movies = load_movies()
            ss = ChunkedSemanticSearch()
            ss.load_or_create_chunk_embeddings(movies)

            results = ss.search_chunks(args.query, limit=args.limit)
            for i, res in enumerate(results, 1):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['document']}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
