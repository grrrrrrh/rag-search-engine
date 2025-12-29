#!/usr/bin/env python3
from __future__ import annotations

import argparse

from lib.multimodal_search import verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    v = subparsers.add_parser("verify_image_embedding", help="Generate and verify an image embedding")
    v.add_argument("image_path", type=str, help="Path to an image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)


if __name__ == "__main__":
    main()
