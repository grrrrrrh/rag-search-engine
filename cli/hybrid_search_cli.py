#!/usr/bin/env python3
import argparse

from lib.hybrid_search import HybridSearch, normalize_scores
from lib.query_enhance import enhance_query_spell, enhance_query_rewrite, enhance_query_expand


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Min-max normalize a list of scores")
    normalize_parser.add_argument("scores", nargs="+", type=float, help="Scores to normalize")

    weighted_parser = subparsers.add_parser("weighted-search", help="Weighted hybrid search (BM25 + semantic)")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 (0..1)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    rrf_parser = subparsers.add_parser("rrf-search", help="RRF hybrid search (BM25 + semantic)")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF k parameter")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results")
    rrf_parser.add_argument("--enhance", choices=["spell", "rewrite", "expand"], default=None, help="Query enhancement")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normed = normalize_scores(args.scores)
            for v in normed:
                print(f"{v:.4f}")

        case "weighted-search":
            hs = HybridSearch()
            results = hs.weighted_search(args.query, alpha=args.alpha, limit=args.limit)
            for r in results:
                print(r["title"])

        case "rrf-search":
            query = args.query
            if args.enhance == "spell":
                enhanced = enhance_query_spell(query)
                print(f"Enhanced query (spell): '{query}' -> '{enhanced}'\n")
                query = enhanced
            elif args.enhance == "rewrite":
                enhanced = enhance_query_rewrite(query)
                print(f"Enhanced query (rewrite): '{query}' -> '{enhanced}'\n")
                print("rewrite")
                query = enhanced

            elif args.enhance == "expand":
                expansion = enhance_query_expand(query)
                enhanced = f"{query} {expansion}".strip()
                print(f"Enhanced query (expand): '{query}' -> '{enhanced}'\n")
                query = enhanced
            hs = HybridSearch()
            results = hs.rrf_search(query, k=args.k, limit=args.limit)

            for i, r in enumerate(results, 1):
                print(f"\n{i}. {r['title']}")
                print(f"   RRF Score: {r['rrf_score']:.3f}")
                print(f"   BM25 Rank: {r.get('bm25_rank')}, Semantic Rank: {r.get('semantic_rank')}")
                snippet = (r.get("document") or "").replace("\n", " ").strip()
                if len(snippet) > 100:
                    snippet = snippet[:100].rstrip() + "..."
                print(f"   {snippet}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
