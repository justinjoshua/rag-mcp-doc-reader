"""MCP server exposing the document corpus as tools.

Run:
    python -m docreader.mcp_server        # stdio transport (default)

Any MCP client (Claude Desktop, Claude Code, etc.) connects and calls
`search_documents` to retrieve grounded context, then synthesizes the answer.
The server itself never calls a model — retrieval only — so it needs no API key.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .config import CONFIG
from .retrieve import retrieve
from .store import get_store

mcp = FastMCP("doc-reader")


@mcp.tool()
def search_documents(query: str, top_k: int = CONFIG.top_k) -> str:
    """Search the indexed document corpus for passages relevant to a question.

    Returns the most relevant chunks with their source file and a similarity
    score. Use the returned passages to answer the user's question and cite the
    source filenames. If nothing relevant is returned, say so rather than
    guessing.

    Args:
        query: A natural-language question or search phrase.
        top_k: How many passages to return (default from config).
    """
    chunks = retrieve(query, top_k)
    if not chunks:
        return "No indexed documents matched. The corpus may be empty — run ingest first."

    blocks = []
    for c in chunks:
        blocks.append(
            f"[source: {c.source} · chunk {c.chunk_index} · score {c.score}]\n{c.text}"
        )
    return "\n\n---\n\n".join(blocks)


@mcp.tool()
def list_sources() -> str:
    """List the distinct document files currently indexed and searchable."""
    sources = get_store().sources()
    if not sources:
        return "No documents indexed yet."
    return f"{len(sources)} document(s) indexed:\n" + "\n".join(f"- {s}" for s in sources)


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
