import argparse

from lib.multimodal_search import image_search_command, verify_image_embedding


def _truncate(s: str, n: int = 95) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "..."


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI (CLIP)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    v = subparsers.add_parser("verify_image_embedding", help="Generate an embedding for an image and print its shape")
    v.add_argument("image_path", type=str, help="Path to an image file")

    s = subparsers.add_parser("image_search", help="Search movies using an image query")
    s.add_argument("image_path", type=str, help="Path to an image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            results = image_search_command(args.image_path, limit=5)
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['title']} (similarity: {r['similarity']:.3f})")
                print(f"   {_truncate(r['description'])}")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
