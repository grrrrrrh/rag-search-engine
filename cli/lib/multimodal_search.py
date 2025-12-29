from __future__ import annotations

from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path: str):
        # PIL loads the image into a format CLIP can process
        img = Image.open(image_path)
        # encode returns an array of embeddings; for one item take [0]
        embedding = self.model.encode([img])[0]
        return embedding


def verify_image_embedding(image_path: str) -> None:
    ms = MultimodalSearch()
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
