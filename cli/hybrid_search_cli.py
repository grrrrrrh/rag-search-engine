#!/usr/bin/env python3
import argparse
import os
from lib.hybrid_search import HybridSearch, normalize_scores
from lib.query_enhance import enhance_query_expand, enhance_query_rewrite, enhance_query_spell, evaluate_results_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    # normalize
    norm = subparsers.add_parser("normalize", help="Normalize a list of scores to 0..1")
    norm.add_argument("scores", type=float, nargs="+", help="Scores")

    # weighted hybrid
    weighted = subparsers.add_parser("weighted-search", help="Weighted hybrid search (BM25 + semantic)")
    weighted.add_argument("query", type=str, help="Search query")
    weighted.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 (0..1)")
    weighted.add_argument("--limit", type=int, default=5, help="Number of results")

    # rrf hybrid
    rrf = subparsers.add_parser("rrf-search", help="RRF hybrid search (BM25 + semantic)")
    rrf.add_argument("query", type=str, help="Search query")
    rrf.add_argument("-k", type=int, default=60, help="RRF k parameter")
    rrf.add_argument("--limit", type=int, default=5, help="Number of results")
    rrf.add_argument("--enhance", choices=["spell", "rewrite", "expand"], default=None, help="Query enhancement")
    rrf.add_argument("--rerank-method", choices=["individual", "batch", "cross_encoder"], default=None)
    rrf.add_argument("--evaluate", action="store_true", help="Evaluate top results with an LLM")
    args = parser.parse_args()


    debug = os.getenv("DEBUG_RAG", "").strip().lower() not in ("", "0", "false")
    match args.command:
        case "normalize":
            normed = normalize_scores(list(args.scores))
            for v in normed:
                print(f"{v:.4f}")

        case "weighted-search":
            hs = HybridSearch()
            results = hs.weighted_search(args.query, alpha=args.alpha, limit=args.limit)
            for r in results:
                print(r["title"])

        case "rrf-search":
            query = args.query

            if debug:
                print(f"[DEBUG] original query: {query!r}")
            if args.enhance == "spell":
                enhanced = enhance_query_spell(query)
                print(f"Enhanced query (spell): '{query}' -> '{enhanced}'\n")
                query = enhanced

            elif args.enhance == "rewrite":
                enhanced = enhance_query_rewrite(query)
                print(f"Enhanced query (rewrite): '{query}' -> '{enhanced}'\n")
                query = enhanced
                print("rewrite")  # boot.dev expects this marker

            elif args.enhance == "expand":
                expansion = enhance_query_expand(query)
                enhanced = (query + " " + expansion).strip()
                print(f"Enhanced query (expand): '{query}' -> '{enhanced}'\n")
                query = enhanced

            # gather 5x results if reranking
            base_limit = int(args.limit)
            rrf_limit = base_limit * 5 if args.rerank_method else base_limit

            if debug:
                print(f"[DEBUG] query after enhancement: {query!r}")

            hs = HybridSearch()
            results = hs.rrf_search(query, k=args.k, limit=rrf_limit)

            if debug:
                print("[DEBUG] results after RRF:")
                for i, r in enumerate(results[:min(len(results), 10)], 1):
                    print(f"[DEBUG]  {i}. {r.get('title')} | rrf={r.get('rrf_score')} | bm25_rank={r.get('bm25_rank')} | semantic_rank={r.get('semantic_rank')}")
            # rerank
            if args.rerank_method == "cross_encoder":
                print(f"Reranking top {rrf_limit} results using cross_encoder method...\n")

                from sentence_transformers import CrossEncoder

                pairs = []
                for doc in results:
                    pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                scores = cross_encoder.predict(pairs)

                for doc, s in zip(results, scores):
                    doc["cross_encoder_score"] = float(s)

                results.sort(key=lambda d: (-d["cross_encoder_score"], d["rrf_rank"]))

            elif args.rerank_method in ("individual", "batch"):
                # deterministic bi-encoder rerank (fast + stable): doc-level embedding similarity
                print(f"Reranking top {rrf_limit} results using {args.rerank_method} method...\n")

                from lib.semantic_search import SemanticSearch

                ss = SemanticSearch()
                # ensure doc embeddings exist
                from lib.search_utils import load_movies
                movies = load_movies()
                doc_emb = ss.load_or_create_embeddings(movies)

                # build id->row index mapping
                id_to_row = {int(m["id"]): i for i, m in enumerate(movies)}

                q = ss.model.encode(query)
                # cosine similarity
                import numpy as np

                qn = np.linalg.norm(q)
                for doc in results:
                    doc_id = int(doc["id"])
                    row = id_to_row.get(doc_id)
                    if row is None:
                        doc["rerank_score"] = 0.0
                        continue
                    v = doc_emb[row]
                    denom = (qn * (np.linalg.norm(v) + 1e-12))
                    doc["rerank_score"] = float(np.dot(q, v) / denom) if denom else 0.0

                results.sort(key=lambda d: (-d["rerank_score"], d["rrf_rank"]))

            if getattr(args, "evaluate", False):
                top = results[:args.limit]
                scores = evaluate_results_llm(query, top)
                print("LLM-evaluate results 0-3")
                for i, (res, score) in enumerate(zip(top, scores), 1):
                    print(f"{i}. {res['title']}: {score}/3")
                print()
            print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):")
            for i, res in enumerate(results[:base_limit], 1):
                print(f"{i}. {res['title']}")
                if args.rerank_method == "cross_encoder":
                    print(f"   Cross Encoder Score: {res.get('cross_encoder_score', 0.0):.3f}")
                elif args.rerank_method in ("individual", "batch"):
                    print(f"   Rerank Rank: {i}")

                print(f"   RRF Score: {res.get('rrf_score', 0.0):.3f}")
                print(f"   BM25 Rank: {res.get('bm25_rank')}, Semantic Rank: {res.get('semantic_rank')}")
                doc_text = (res.get("document") or "").replace("\n", " ").strip()
                if len(doc_text) > 90:
                    doc_text = doc_text[:90].rstrip() + "..."
                print(f"   {doc_text}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
