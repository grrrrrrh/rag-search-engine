from cli.lib.search_utils import load_movies
from cli.lib.hybrid_search import HybridSearch

def run_case(alpha: float, limit: int = 25) -> None:
    query = "British Bear"
    movies = load_movies()
    hs = HybridSearch(movies)

    results = hs.weighted_search(query, alpha=alpha, limit=limit)

    print("\n" + "=" * 80)
    print(f"QUERY: {query!r}  alpha={alpha}  limit={limit}")
    print("=" * 80)
    for i, r in enumerate(results, 1):
        doc_id = r.get("id")
        title = r.get("title", "")
        hybrid = r.get("hybrid", 0.0)
        bm25 = r.get("bm25", 0.0)
        sem = r.get("semantic", 0.0)
        print(f"{i:2d}. ({doc_id}) {title}")
        print(f"    hybrid={hybrid:.6f}  bm25={bm25:.6f}  semantic={sem:.6f}")

def main() -> None:
    for a in (0.5, 0.2, 0.8):
        run_case(a, 25)

if __name__ == "__main__":
    main()
