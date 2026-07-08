"""ChromaDB persistent vector store wrapper.

Embeddings are always supplied by the caller (our Embedder) — Chroma's built-in
embedding function is intentionally not used, so ingest and query can never drift
onto different models.

The collection handle is resolved by name on every operation rather than cached.
That keeps a long-running process (e.g. the web server) resilient to an external
`ingest --reset` in another process, which deletes and recreates the collection:
a cached handle would point at the deleted collection and raise NotFoundError.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import chromadb

from .config import CONFIG


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    score: float = 0.0  # 1 - distance; higher is more similar


class Store:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=CONFIG.db_path_abs)

    def _col(self):
        return self._client.get_or_create_collection(
            CONFIG.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids, embeddings, documents, metadatas) -> None:
        # upsert makes re-ingesting the same content a no-op instead of an error
        self._col().upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[Chunk]:
        res = self._col().query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
        return [
            Chunk(
                text=doc,
                source=meta.get("source", "unknown"),
                chunk_index=int(meta.get("chunk_index", 0)),
                score=round(1.0 - dist, 4),
            )
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def all_chunks(self) -> list[Chunk]:
        """Every stored chunk (used for the keyword pass of hybrid retrieval)."""
        res = self._col().get(include=["documents", "metadatas"])
        return [
            Chunk(
                text=doc,
                source=meta.get("source", "unknown"),
                chunk_index=int(meta.get("chunk_index", 0)),
            )
            for doc, meta in zip(res["documents"], res["metadatas"])
        ]

    def sources(self) -> list[str]:
        res = self._col().get(include=["metadatas"])
        return sorted({m.get("source", "unknown") for m in res["metadatas"]})

    def delete_source(self, source: str) -> int:
        """Remove all chunks belonging to one source document. Returns count removed."""
        col = self._col()
        before = col.count()
        col.delete(where={"source": source})
        return before - col.count()

    def count(self) -> int:
        return self._col().count()

    def reset(self) -> None:
        try:
            self._client.delete_collection(CONFIG.collection_name)
        except Exception:  # noqa: BLE001 - already absent is fine
            pass
        self._col()  # recreate immediately


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store()
