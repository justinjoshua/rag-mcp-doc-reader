"""End-to-end smoke test — requires NO API key (forces the extractive backend).

Covers: ingest -> hybrid retrieval -> MCP tools -> streaming -> single-file
ingest (the upload path). Run:

    PYTHONPATH=src ./.venv/Scripts/python.exe scripts/e2e_test.py
"""
import dataclasses
import sys
import tempfile
from pathlib import Path

import docreader.answer as A
from docreader import ingest as ingest_mod
from docreader import mcp_server
from docreader.config import CONFIG
from docreader.retrieve import retrieve
from docreader.store import get_store

# Force extractive so the test never calls a paid/remote model.
A.CONFIG = dataclasses.replace(A.CONFIG, answer_backend="extractive")

print("== 1. Ingest ==")
ingest_mod.ingest(CONFIG.docs_dir, reset=True)
assert get_store().count() > 0
print("indexed chunks:", get_store().count())

print("\n== 2. Hybrid retrieval (keyword term should surface exact match) ==")
hits = retrieve("Globex contract", top_k=3)
assert any("Globex" in c.text for c in hits), "hybrid retrieval missed the keyword hit"
print("PASS: 'Globex' surfaced via hybrid retrieval; top score", hits[0].score)

print("\n== 3. MCP search_documents ==")
assert "139" in mcp_server.search_documents("Q3 revenue", top_k=3)
print("PASS: MCP tool returned the Q3 figure")

print("\n== 4. Streaming (extractive) yields deltas then done ==")
evs = list(A.answer_stream("What is the FY2026 outlook?", top_k=3))
assert evs[-1]["type"] == "done", "stream must end with a done event"
assert any(e["type"] == "delta" for e in evs), "no delta events"
assert evs[-1]["citations"], "done event must carry passage citations"
print(f"PASS: {sum(e['type']=='delta' for e in evs)} delta(s) + done with "
      f"{len(evs[-1]['citations'])} citations")

print("\n== 5. answer_question (non-streaming wrapper) ==")
res = A.answer_question("outlook", top_k=3)
assert res.backend == "extractive" and res.citations
print("PASS: non-streaming wrapper returns a result")

print("\n== 6. Follow-up with history is accepted ==")
res2 = A.answer_question("and the Q3 number?",
                         history=[{"role": "user", "content": "what is the outlook"},
                                  {"role": "assistant", "content": res.answer}], top_k=3)
assert res2.found_context
print("PASS: history threaded through without error")

print("\n== 7. Single-file ingest (browser-upload path) ==")
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "note.md"
    p.write_text("# Note\n\nThe secret passphrase is orange-elephant-42.", encoding="utf-8")
    added = ingest_mod.ingest_file(p, source="note.md")
    assert added > 0
    assert "note.md" in get_store().sources()
    assert any("orange-elephant-42" in c.text for c in retrieve("secret passphrase", top_k=3))
print("PASS: uploaded file ingested and retrievable")

print("\n== 8. Delete a document from the index ==")
removed = get_store().delete_source("note.md")
assert removed > 0
assert "note.md" not in get_store().sources()
print(f"PASS: delete_source removed {removed} chunk(s); source gone")

print("\n== 9. Citations carry a numeric relevance score ==")
res3 = A.answer_question("Q3 revenue", top_k=2)
assert res3.citations and all(isinstance(c.score, (int, float)) for c in res3.citations)
print(f"PASS: top citation score = {res3.citations[0].score}")

print("\n== 10. Re-ranking sharpens relevance ==")
assert CONFIG.rerank, "rerank should be enabled by default"
top = retrieve("Q3 revenue", top_k=2)
assert top[0].chunk_index == 0 and top[0].score > 0.5, (top[0].chunk_index, top[0].score)
print(f"PASS: reranker placed chunk 0 first with score {top[0].score}")

print("\nALL CHECKS PASSED")
sys.exit(0)
