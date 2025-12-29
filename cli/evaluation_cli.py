#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from lib.hybrid_search import HybridSearch


def _project_root() -> Path:
    # cli/evaluation_cli.py -> repo root
    return Path(__file__).resolve().parents[1]


def load_golden_dataset() -> dict:
    path = _project_root() / "data" / "golden_dataset.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def precision_recall_f1_at_k(
    retrieved_titles: list[str], relevant_titles: list[str], k: int
) -> tuple[float, float, float]:
    if k <= 0:
        return 0.0, 0.0, 0.0

    retrieved_set = set(retrieved_titles[:k])
    relevant_set = set(relevant_titles)

    hits = len(retrieved_set & relevant_set)

    precision = hits / float(k)
    recall = (hits / float(len(relevant_set))) if relevant_set else 0.0

    denom = precision + recall
    f1 = (2.0 * precision * recall / denom) if denom > 0.0 else 0.0

    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=5, help="k for Precision@k/Recall@k/F1")
    args = parser.parse_args()

    k = args.limit
    data = load_golden_dataset()
    test_cases = data.get("test_cases", [])

    hs = HybridSearch()

    debug = os.getenv("DEBUG_RAG", "").strip() not in ("", "0", "false", "False")

    print(f"k={k}\n")

    for case in test_cases:
        query = case.get("query", "")
        relevant_docs = case.get("relevant_docs", [])

        results = hs.rrf_search(query, k=60, limit=k)
        retrieved_titles = [r.get("title", "") for r in results if r.get("title")]

        precision, recall, f1 = precision_recall_f1_at_k(retrieved_titles, relevant_docs, k)

        print(f"- Query: {query}")
        print(f"  - Precision@{k}: {precision:.4f}")
        print(f"  - Recall@{k}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_docs)}")

        if debug:
            hits = sorted(set(retrieved_titles[:k]) & set(relevant_docs))
            print(f"  - DEBUG hits({len(hits)}): {', '.join(hits)}")

        print()


if __name__ == "__main__":
    main()
