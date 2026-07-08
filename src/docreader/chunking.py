"""Recursive character chunker with overlap — dependency-free.

Splits on the largest natural boundary that keeps chunks under CHUNK_SIZE
(paragraphs -> lines -> sentences -> words), then stitches pieces back up to the
target size with CHUNK_OVERLAP characters carried between neighbours so context
isn't cut mid-thought.
"""
from __future__ import annotations

import re

from .config import CONFIG

# Ordered from coarsest to finest separator.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, separators: list[str], size: int) -> list[str]:
    if len(text) <= size or not separators:
        return [text]
    sep, *rest = separators
    parts = text.split(sep)
    out: list[str] = []
    for part in parts:
        piece = part + sep
        if len(piece) <= size:
            out.append(piece)
        else:
            out.extend(_split(piece, rest, size))
    return out


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Return a list of overlapping chunks. Empty/whitespace input -> []."""
    size = size or CONFIG.chunk_size
    overlap = overlap or CONFIG.chunk_overlap
    text = text.strip()
    if not text:
        return []

    pieces = _split(text, _SEPARATORS, size)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) <= size:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            # carry the tail of the previous chunk forward as overlap
            tail = current[-overlap:] if overlap else ""
            current = tail + piece
    if current.strip():
        chunks.append(current.strip())

    # collapse runs of whitespace introduced by joining
    return [re.sub(r"[ \t]+", " ", c) for c in chunks]
