"""Standalone RAG answerer (CLI over the shared answer core).

Usage:
    python -m docreader.ask "What does the report say about Q3 revenue?"

Prints a grounded answer plus the source passages it used. The answer backend
(gemini / extractive / ollama) is set by ANSWER_BACKEND in .env.
"""
from __future__ import annotations

import argparse
import sys

from .answer import answer_question

# On Windows the default console codepage (cp1252) raises UnicodeEncodeError
# when printing characters it can't represent (em dashes, curly quotes, etc.).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask a question against your indexed docs.")
    ap.add_argument("question", help="The question to answer.")
    ap.add_argument("--top-k", type=int, default=None, help="Chunks to retrieve.")
    args = ap.parse_args()

    result = answer_question(args.question, top_k=args.top_k)

    print("\n=== Answer ===\n")
    print(result.answer)

    if result.citations:
        print("\n=== Sources ===\n")
        for c in result.citations:
            snippet = c.text.replace("\n", " ")[:160]
            print(f'- {c.title}: "{snippet}..."')


if __name__ == "__main__":
    main()
