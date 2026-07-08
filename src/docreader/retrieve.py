"""Retrieval core, shared by every interface (CLI, web, MCP).

Pipeline:
  1. Candidate generation — CONFIG.retrieval_mode:
       "vector": pure embedding similarity.
       "hybrid": fuse vector similarity with a keyword pass via Reciprocal Rank
         Fusion (RRF). Hybrid does better on exact terms — names, IDs, numbers.
  2. Re-ranking (CONFIG.rerank) — a cross-encoder re-scores the candidates so
     the passage most relevant to the query ends up first. The chunk `.score`
     shown in the UI becomes the cross-encoder relevance (0–1).
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

from .config import CONFIG
from .embeddings import get_embedder
from .store import Chunk, get_store

_WORD = re.compile(r"\w+")
_RRF_K = 60


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _keyword_rank(query: str, chunks: list[Chunk]) -> list[Chunk]:
    terms = set(_tokens(query))
    if not terms:
        return []
    scored = []
    for c in chunks:
        hits = sum(1 for t in _tokens(c.text) if t in terms)
        if hits:
            scored.append((hits, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


def _key(c: Chunk) -> tuple[str, int]:
    return (c.source, c.chunk_index)


def _rrf(rankings: list[list[Chunk]], n: int) -> list[Chunk]:
    scores: dict[tuple[str, int], float] = {}
    best: dict[tuple[str, int], Chunk] = {}
    for ranking in rankings:
        for rank, c in enumerate(ranking):
            k = _key(c)
            scores[k] = scores.get(k, 0.0) + 1.0 / (_RRF_K + rank)
            if k not in best or (c.score and not best[k].score):
                best[k] = c
    ordered = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [best[k] for k in ordered[:n]]


def _candidates(query: str, n: int) -> list[Chunk]:
    store = get_store()
    query_vec = get_embedder().embed_query(query)
    if CONFIG.retrieval_mode.lower() != "hybrid":
        return store.query(query_vec, n)
    vector_hits = store.query(query_vec, n)
    keyword_hits = _keyword_rank(query, store.all_chunks())[:n]
    if not keyword_hits:
        return vector_hits[:n]
    return _rrf([vector_hits, keyword_hits], n)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(CONFIG.rerank_model)


def _rerank(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
    scores = _reranker().predict([(query, c.text) for c in chunks])
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order[:top_k]:
        c = chunks[i]
        c.score = round(1.0 / (1.0 + math.exp(-float(scores[i]))), 4)  # sigmoid -> 0..1
        out.append(c)
    return out


def retrieve(query: str, top_k: int | None = None) -> list[Chunk]:
    top_k = top_k or CONFIG.top_k
    # Pull a wider candidate set when reranking so the cross-encoder has options.
    n = max(top_k * 4, 20) if CONFIG.rerank else top_k
    candidates = _candidates(query, n)
    if CONFIG.rerank and candidates:
        return _rerank(query, candidates, top_k)
    return candidates[:top_k]
