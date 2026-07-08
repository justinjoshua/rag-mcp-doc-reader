"""Shared RAG answering core: streaming, multi-turn, pluggable backends.

Retrieval is always local (embeddings + ChromaDB). The answer step is chosen by
CONFIG.answer_backend:

    gemini      Google Gemini API    -> needs GEMINI_API_KEY (primary)
    extractive  no LLM, no key       -> returns the ranked passages (fallback)
    ollama      local LLM (Ollama)   -> free, offline, no key

`answer_stream()` is the primitive: it yields {"type": "delta", "text": ...}
events as the answer is produced, then a final {"type": "done", ...} carrying
citations / sources / backend. `answer_question()` wraps it for non-streaming
callers. Both accept optional conversation `history` for follow-up questions.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from .config import CONFIG
from .retrieve import retrieve
from .store import Chunk

SYSTEM = (
    "You answer questions using the provided document excerpts and the "
    "conversation so far. Prefer the excerpts; cite the source filenames you "
    "use. If the excerpts do not contain the answer, say so plainly instead of "
    "guessing. Be concise."
)

Event = dict  # {"type": "delta"|"done"|"error", ...}
History = list[dict]  # [{"role": "user"|"assistant", "content": str}, ...]

# Runtime backend override. CONFIG is frozen (set from .env at startup); this lets
# the UI flip between "gemini" and "extractive" live without a restart. None => use
# whatever CONFIG.answer_backend was configured with.
_backend_override: str | None = None

# Best-effort, in-process usage tally so the UI can show how hard the Gemini free
# tier is being pushed this session. Not Google's official number — resets on restart.
_gemini_calls = 0
_gemini_quota_errors = 0


def gemini_stats() -> dict:
    return {"calls": _gemini_calls, "quota_errors": _gemini_quota_errors}


def current_backend() -> str:
    return (_backend_override or CONFIG.answer_backend).lower()


def set_backend(name: str) -> str:
    """Switch the active answer backend at runtime. Returns the backend now in use."""
    global _backend_override
    name = (name or "").lower().strip()
    if name not in _STREAMS:
        raise ValueError(f"Unknown backend {name!r}. Use one of: {', '.join(_STREAMS)}.")
    _backend_override = name
    return name


@dataclass
class Citation:
    title: str
    text: str
    score: float = 0.0


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    found_context: bool = True
    backend: str = ""
    error: bool = False


def _passage_citations(chunks: list[Chunk]) -> list[Citation]:
    return [Citation(f"{c.source} #{c.chunk_index}", c.text, c.score) for c in chunks]


def _distinct_sources(chunks: list[Chunk]) -> list[str]:
    return sorted({c.source for c in chunks})


def _history_block(history: History) -> str:
    if not history:
        return ""
    lines = [
        f"{'User' if t.get('role') == 'user' else 'Assistant'}: {t.get('content', '')}"
        for t in history
    ]
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


def _prompt(question: str, chunks: list[Chunk], history: History) -> str:
    context = "\n\n".join(f"[{c.source} #{c.chunk_index}]\n{c.text}" for c in chunks)
    return f"{_history_block(history)}Document excerpts:\n\n{context}\n\nQuestion: {question}"


# --- streaming backends (yield {"type": "delta"|"error", "text": ...}) -------

# Leading bullet / numbering markers to strip when presenting a line.
_MARKER = re.compile(r"^\s*(?:[-*•‣◦→➤▪·]|\d+[.)]|[①-⑳]|\([a-z0-9]+\))\s*")


def _units(text: str) -> list[str]:
    """Break a chunk into candidate 'points' — one per line/sentence — for scoring."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line):
            part = part.strip()
            if len(part) >= 15:  # drop fragments / OCR noise
                out.append(part)
    return out


def _content_weight(u: str) -> float:
    """Favor explanatory lines over bare headings when ranking. A line like
    'Change control :-' is a label; 'It is a process to manage changes...' is
    content. We down-weight the former so short queries surface real answers."""
    body = _MARKER.sub("", u).strip()
    # A bare label: ends with ':' or ':-' and has little/nothing after the colon.
    if re.search(r":-?\s*$", body):
        return 0.45
    w = 1.0
    if re.search(r":-?\s+\S", body):   # "label :- actual explanation"
        w += 0.2
    if len(body) >= 45:                 # substantial content
        w += 0.2
    if len(body) < 22:                  # thin fragment
        w -= 0.2
    return w


def _extractive_answer(question: str, chunks: list[Chunk],
                       max_points: int = 6) -> list[str]:
    """Query-focused extractive summary, fully local (no LLM): score every point
    against the question with the embedding model and keep the best-matching ones."""
    from .embeddings import get_embedder

    # Only mine lines from the chunks that actually scored well. Retrieval /
    # reranking already knows which passages are on-topic (e.g. a 0.94 chunk vs
    # 0.00 filler), so restrict to those — otherwise unrelated lines leak in.
    top_score = max((c.score for c in chunks), default=0.0)
    if top_score > 0:
        # Tight band: when one chunk clearly dominates, answer from it alone so an
        # off-topic-but-decently-scored chunk can't inject noise.
        keep = {i for i, c in enumerate(chunks) if c.score >= 0.8 * top_score}
    else:
        keep = set(range(min(2, len(chunks))))

    seen: set[str] = set()
    cands: list[tuple[int, int, str]] = []
    for ci, c in enumerate(chunks):
        if ci not in keep:
            continue
        for ui, u in enumerate(_units(c.text)):
            key = re.sub(r"\W+", "", u.lower())[:80]
            if not key or key in seen:  # dedup overlapping chunks
                continue
            seen.add(key)
            cands.append((ci, ui, u))
    if not cands:
        return []

    emb = get_embedder()
    qv = emb.embed_query(question)
    uvs = emb.embed_documents([u for _, _, u in cands])
    # embeddings are normalized -> cosine similarity is just the dot product.
    # Rank by relevance * content-richness, but gate on raw relevance so a
    # wordy-but-irrelevant line can't sneak in.
    scored = [
        (sum(a * b for a, b in zip(qv, uv)) * _content_weight(u),  # ranking score
         sum(a * b for a, b in zip(qv, uv)),                       # raw relevance
         ci, ui, u)
        for (ci, ui, u), uv in zip(cands, uvs)
    ]
    scored.sort(key=lambda x: -x[0])
    top = [s for s in scored[:max_points] if s[1] > 0.2] or scored[:3]
    top.sort(key=lambda s: (s[2], s[3]))  # restore document order for readability
    return [_MARKER.sub("", u).strip() for _, _, _, _, u in top]


def _to_sentence(p: str) -> str:
    """Normalize an extracted line into a standalone sentence for prose output."""
    p = re.sub(r"\s*:-\s*", ": ", p).strip()   # "label :- text" -> "label: text"
    p = p.rstrip(" ;:-,")
    if not p:
        return ""
    p = p[0].upper() + p[1:]
    if p[-1] not in ".!?":
        p += "."
    return p


def _extractive_stream(question, chunks, history) -> Iterator[Event]:
    # No LLM, no API: build a real query-focused answer from the most relevant
    # lines in the retrieved passages, stitched into a readable paragraph.
    points = _extractive_answer(question, chunks)
    if not points:
        yield {"type": "delta", "text":
               "I couldn't find anything relevant to that in your documents."}
        return
    para = " ".join(s for s in (_to_sentence(p) for p in points) if s)
    yield {"type": "delta", "text": para}
    yield {"type": "delta", "text":
           "\n\n_Local answer — assembled from the most relevant lines in your notes "
           "(no AI). Switch to Gemini for a fuller written explanation._"}


def _gemini_stream(question, chunks, history) -> Iterator[Event]:
    global _gemini_calls, _gemini_quota_errors
    if not CONFIG.gemini_api_key:
        yield {"type": "error", "text": "No Gemini API key configured. Set "
               "GEMINI_API_KEY in .env (get one at https://aistudio.google.com/apikey)."}
        return
    _gemini_calls += 1
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=CONFIG.gemini_api_key)
        stream = client.models.generate_content_stream(
            model=CONFIG.gemini_model,
            contents=_prompt(question, chunks, history),
            config=types.GenerateContentConfig(system_instruction=SYSTEM),
        )
        for ev in stream:
            if getattr(ev, "text", None):
                yield {"type": "delta", "text": ev.text}
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            _gemini_quota_errors += 1
            yield {"type": "error", "text": (
                "⚠️ Gemini quota / rate limit reached. Free-tier limits reset "
                "after a short wait (per-minute) or daily. Options: wait and "
                "retry, set ANSWER_BACKEND=extractive in .env to browse passages "
                "without the LLM, or enable billing on your Google AI Studio key. "
                "The most relevant passages are shown under Sources below.")}
        else:
            yield {"type": "error", "text": f"Gemini request failed: {e}"}


def _ollama_stream(question, chunks, history) -> Iterator[Event]:
    import json

    import requests

    try:
        with requests.post(
            f"{CONFIG.ollama_url}/api/chat",
            json={
                "model": CONFIG.ollama_model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": _prompt(question, chunks, history)},
                ],
                "stream": True,
            },
            stream=True,
            timeout=120,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                tok = json.loads(line).get("message", {}).get("content", "")
                if tok:
                    yield {"type": "delta", "text": tok}
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "text": f"Ollama request failed: {e}. Is Ollama "
               f"running and is the model '{CONFIG.ollama_model}' pulled?"}


_STREAMS = {"gemini": _gemini_stream, "extractive": _extractive_stream, "ollama": _ollama_stream}


# --- public API -------------------------------------------------------------

def answer_stream(question: str, history: History | None = None,
                  top_k: int | None = None) -> Iterator[Event]:
    history = history or []
    backend = current_backend()
    chunks = retrieve(question, top_k)

    if not chunks:
        yield {"type": "delta", "text": "No indexed documents matched. Have you "
               "ingested any documents yet?"}
        yield {"type": "done", "backend": backend, "error": False,
               "found_context": False, "citations": [], "sources": []}
        return

    gen = _STREAMS.get(backend)
    if gen is None:
        yield {"type": "delta", "text": f"Unknown ANSWER_BACKEND: {backend!r}. "
               f"Use one of: {', '.join(_STREAMS)}."}
        yield {"type": "done", "backend": backend, "error": True,
               "found_context": True, "citations": [], "sources": []}
        return

    error = False
    for ev in gen(question, chunks, history):
        if ev["type"] == "error":
            error = True
            yield {"type": "delta", "text": ev["text"]}
        else:
            yield ev

    cites = _passage_citations(chunks)
    yield {
        "type": "done",
        "backend": backend,
        "error": error,
        "found_context": True,
        "citations": [{"title": c.title, "text": c.text, "score": c.score} for c in cites],
        "sources": _distinct_sources(chunks),
        "stats": gemini_stats(),
    }


def answer_question(question: str, history: History | None = None,
                    top_k: int | None = None) -> AnswerResult:
    parts: list[str] = []
    done: Event | None = None
    for ev in answer_stream(question, history, top_k):
        if ev["type"] == "delta":
            parts.append(ev["text"])
        elif ev["type"] == "done":
            done = ev
    done = done or {}
    return AnswerResult(
        answer="".join(parts).strip(),
        citations=[Citation(c["title"], c["text"], c.get("score", 0.0))
                   for c in done.get("citations", [])],
        sources=done.get("sources", []),
        found_context=done.get("found_context", False),
        backend=done.get("backend", ""),
        error=done.get("error", False),
    )
