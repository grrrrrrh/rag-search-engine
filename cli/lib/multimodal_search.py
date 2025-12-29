from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# Reuse your existing dataset loader (used elsewhere in the course project)
from lib.semantic_search import load_movies


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _ensure_rgb(img: Image.Image) -> Image.Image:
    # CLIP models typically expect RGB
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


class MultimodalSearch:
    """
    Uses a CLIP-style SentenceTransformer model that embeds BOTH text and images
    into the same vector space.
    """

    def __init__(self, documents: list[dict[str, Any]] | None = None, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

        self.documents: list[dict[str, Any]] = documents or []
        self.texts: list[str] = [
            f"{doc['title']}: {doc['description']}"
            for doc in self.documents
        ]

        if self.texts:
            emb = self.model.encode(self.texts, show_progress_bar=True)
            self.text_embeddings = np.asarray(emb, dtype=np.float32)
        else:
            # (0, 0) makes it obvious we have no embeddings
            self.text_embeddings = np.empty((0, 0), dtype=np.float32)

    def embed_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path)
        img = _ensure_rgb(img)
        emb = self.model.encode([img])
        return np.asarray(emb[0], dtype=np.float32)

    def search_with_image(self, image_path: str, limit: int = 5) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        if len(self.documents) == 0 or self.text_embeddings.size == 0:
            return []

        image_emb = self.embed_image(image_path)

        scored: list[dict[str, Any]] = []
        for i, (doc, text_emb) in enumerate(zip(self.documents, self.text_embeddings)):
            score = _cosine_similarity(text_emb, image_emb)
            scored.append(
                {
                    "doc_id": i,
                    "title": doc.get("title", ""),
                    "description": doc.get("description", ""),
                    "similarity": score,
                }
            )

        scored.sort(key=lambda r: r["similarity"], reverse=True)
        return scored[:limit]


def verify_image_embedding(image_path: str) -> None:
    ms = MultimodalSearch()
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str, limit: int = 5) -> list[dict[str, Any]]:
    movies = load_movies()
    ms = MultimodalSearch(documents=movies)
    return ms.search_with_image(image_path, limit=limit)
