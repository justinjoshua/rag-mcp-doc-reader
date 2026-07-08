"""Pluggable embedding provider.

Default is a local sentence-transformers model: no API key, no cost, works
offline, keeps your documents on your own infrastructure. Set
EMBEDDING_PROVIDER=voyage (and VOYAGE_API_KEY) to switch to hosted embeddings
with no other code changes.

The same provider must be used for ingest and query — mixing models produces
meaningless similarity scores.
"""
from __future__ import annotations

from functools import lru_cache

from .config import CONFIG


class Embedder:
    """Encodes text into vectors. `documents` and `query` are distinguished
    because some providers (e.g. Voyage) embed them differently."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


class VoyageEmbedder(Embedder):
    def __init__(self, model_name: str):
        import voyageai

        self.client = voyageai.Client()  # reads VOYAGE_API_KEY
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(texts, model=self.model_name, input_type="document").embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed([text], model=self.model_name, input_type="query").embeddings[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    provider = CONFIG.embedding_provider.lower()
    if provider == "local":
        return LocalEmbedder(CONFIG.embedding_model)
    if provider == "voyage":
        return VoyageEmbedder(CONFIG.embedding_model)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r} (expected 'local' or 'voyage')")
