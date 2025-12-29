import argparse

from lib.augmented_generation import answer_question, answer_with_citations, summarize_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize", help="Summarize results for a query")
    summarize.add_argument("query", type=str, help="Search query")
    summarize.add_argument("--limit", type=int, default=5, help="Number of retrieved docs")

    citations = subparsers.add_parser("citations", help="Answer with citations to retrieved sources")
    citations.add_argument("query", type=str, help="Search query")
    citations.add_argument("--limit", type=int, default=5, help="Number of retrieved docs")

    question = subparsers.add_parser("question", help="Answer a natural-language question using retrieved sources")
    question.add_argument("query", type=str, help="Question text")
    question.add_argument("--limit", type=int, default=5, help="Number of retrieved docs")

    args = parser.parse_args()

    if args.command == "summarize":
        summarize_query(args.query, limit=args.limit)
    elif args.command == "citations":
        answer_with_citations(args.query, limit=args.limit)
    elif args.command == "question":
        answer_question(args.query, limit=args.limit)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
