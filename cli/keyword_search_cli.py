#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path

# allow imports from project root when running cli/ script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inverted_index import InvertedIndex
from bm25 import bm25_idf_command, bm25_tf_command
from search_utils import tokenize, BM25_K1, BM25_B


def load_movies() -> list[dict]:
    data_path = Path("data") / "movies.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    return data.get("movies", [])


def load_index_or_exit() -> InvertedIndex:
    try:
        return InvertedIndex.load()
    except FileNotFoundError:
        print("Error: index not found. Run `build` first.")
        sys.exit(1)


def one_token(term: str) -> str:
    toks = tokenize(term)
    if len(toks) != 1:
        raise ValueError("Term must tokenize to exactly one token.")
    return toks[0]


def compute_idf(idx: InvertedIndex, raw_term: str) -> float:
    tok = one_token(raw_term)
    total_doc_count = len(idx.docmap)
    term_match_doc_count = len(idx.index.get(tok, set()))
    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build and cache the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using keyword search")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to lookup")

    idf_parser = subparsers.add_parser("idf", help="Compute inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to compute IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Compute TF-IDF for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to compute TF-IDF for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Max results (default 5)")

    args = parser.parse_args()

    match args.command:
        case "build":
            idx = InvertedIndex()
            idx.build(load_movies(), tokenize)
            idx.save()

        case "search":
            print(f"Searching for: {args.query}")
            idx = load_index_or_exit()

            results: list[int] = []
            seen: set[int] = set()

            for qt in tokenize(args.query):
                for doc_id in idx.get_documents(qt):
                    if doc_id in seen:
                        continue
                    seen.add(doc_id)
                    results.append(doc_id)
                    if len(results) >= 5:
                        break
                if len(results) >= 5:
                    break

            for i, doc_id in enumerate(results, start=1):
                title = idx.docmap.get(doc_id, {}).get("title", "")
                print(f"{i}. {title} (ID: {doc_id})")

        case "tf":
            idx = load_index_or_exit()
            print(idx.get_tf(args.doc_id, args.term))

        case "idf":
            idx = load_index_or_exit()
            idf = compute_idf(idx, args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            idx = load_index_or_exit()
            tf = idx.get_tf(args.doc_id, args.term)
            idf = compute_idf(idx, args.term)
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, k1=args.k1, b=args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            idx = load_index_or_exit()
            results = idx.bm25_search(args.query, limit=args.limit)

            for i, (doc_id, score) in enumerate(results, start=1):
                title = idx.docmap.get(doc_id, {}).get("title", "")
                print(f"{i}. ({doc_id}) {title} - Score: {score:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
