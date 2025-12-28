#!/usr/bin/env python3

import argparse

from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch, normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # normalize
    norm_parser = subparsers.add_parser("normalize", help="Normalize scores to [0,1]")
    norm_parser.add_argument("scores", nargs="+", type=float, help="Scores")

    # weighted-search
    w_parser = subparsers.add_parser("weighted-search", help="Hybrid search (weighted BM25 + semantic)")
    w_parser.add_argument("query", type=str, help="Search query")
    w_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 (0..1)")
    w_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    # rrf-search
    rrf_parser = subparsers.add_parser("rrf-search", help="Hybrid search using Reciprocal Rank Fusion (RRF)")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF k parameter")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            vals = normalize(args.scores)
            for v in vals:
                print(f"{v:.4f}")

        case "weighted-search":
            movies = load_movies()
            hs = HybridSearch(movies)
            results = hs.weighted_search(args.query, alpha=args.alpha, limit=args.limit)
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['title']}")

        case "rrf-search":
            movies = load_movies()
            hs = HybridSearch(movies)
            results = hs.rrf_search(args.query, k=args.k, limit=args.limit)

            for i, r in enumerate(results, 1):
                bm25_rank = r["bm25_rank"] if r["bm25_rank"] is not None else 0
                sem_rank = r["semantic_rank"] if r["semantic_rank"] is not None else 0

                print(f"\n{i}. {r['title']}")
                print(f"   RRF Score: {r['rrf']:.3f}")
                print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
                print(f"   {r['document']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
