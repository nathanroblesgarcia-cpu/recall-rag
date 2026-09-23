"""Recall MCP server.

Exposes Recall as Model Context Protocol (MCP) tools, so an AI coding assistant
(Claude Code, Claude Desktop, or any MCP client) can search your notes and add
new ones directly instead of shelling out to python. Retrieval runs locally;
nothing leaves the machine unless you set GEN_BACKEND = "anthropic".

It imports the app's real rag/ingest modules, so results match the web page
exactly.

Register it with Claude Code:

    claude mcp add recall -- <path-to>/venv/Scripts/python.exe <path-to>/mcp_server.py
"""

import sys
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Put this folder on the path so we import the app's own modules (config resolves
# its paths via __file__, so the client's working directory does not matter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Recall's modules (config, ingest, rag, fastembed) are slow to import cold,
# which can blow an MCP client's connect timeout (30s in Claude Code) when several
# servers start at once. So they are imported lazily, after the handshake, via
# _load().
#
# The native-extension packages MUST still be imported here, on the main thread,
# before mcp.run(): on Windows, loading a DLL (numpy/onnxruntime) from a worker
# thread while the stdio reader thread blocks on stdin deadlocks forever. They are
# cheap; the slow pure-Python part is what stays lazy.
import numpy  # noqa: E402,F401
import onnxruntime  # noqa: E402,F401
import sqlite3  # noqa: E402,F401
import tokenizers  # noqa: E402,F401

config = ingest = rag = TextEmbedding = None
_LOAD_LOCK = threading.Lock()


def _load():
    global config, ingest, rag, TextEmbedding
    with _LOAD_LOCK:
        if rag is None:
            import config as _config
            import ingest as _ingest
            import rag as _rag
            from fastembed import TextEmbedding as _TE

            config, ingest, TextEmbedding = _config, _ingest, _TE
            rag = _rag  # set last: it is the "loaded" flag


mcp = FastMCP(
    "recall",
    instructions=(
        "Search and feed Recall, a local RAG over the user's own notes. Check "
        "Recall before answering from assumption about their projects, people, "
        "decisions, or past work.\n"
        "\n"
        "Default to recall_search: retrieval only (no local model), fast, and it "
        "returns the actual note chunks for you to read and answer from. Use "
        "recall_ask only when you want Recall's own local model to write the "
        "answer. Use recall_add_note to save a new fact, recall_reindex after "
        "notes were edited on disk, recall_areas to list the area filters. Notes "
        "can be stale, so treat results as leads to verify."
    ),
)

# One embedder, created lazily and reused (loading it is the slow part).
_EMBEDDER = None


def _embedder():
    global _EMBEDDER
    _load()
    if _EMBEDDER is None:
        _EMBEDDER = TextEmbedding(model_name=config.EMBED_MODEL)
    return _EMBEDDER


@mcp.tool()
def recall_search(question: str, top_k: int = 8, area: str = "") -> str:
    """Semantic search over Recall. Returns the closest note chunks with their
    source and score. Fast: retrieval only, no local model. Read the chunks and
    write the answer yourself.

    area: optional area filter (see recall_areas). Empty = search everything.
    """
    _load()
    hits = rag.retrieve(question, top_k=top_k, area=(area or None))
    if not hits:
        return "No matching notes found in Recall."
    lines = []
    for i, h in enumerate(hits, 1):
        snippet = " ".join(h["text"].split())
        if len(snippet) > 600:
            snippet = snippet[:600] + " ..."
        lines.append(f"[{i}] {h['source']} (score {h['score']:.3f})\n{snippet}")
    return "\n\n".join(lines)


@mcp.tool()
def recall_ask(question: str, area: str = "", smarter: bool = False) -> str:
    """Ask Recall a question and get an answer written by Recall's OWN model
    (local by default), with the source notes listed. Prefer recall_search when
    you can write the answer yourself.

    smarter: False = the default answer model; True = the larger judge model
    (slower, better at pulling facts cleanly).
    area: optional area filter (see recall_areas). Empty = search everything.
    """
    _load()
    model = config.JUDGE_MODEL if smarter else None
    result = rag.answer_question(question, model=model, area=(area or None))
    answer = result.get("answer", "").strip()
    sources = []
    seen = set()
    for s in result.get("sources", []):
        src = s.get("source") if isinstance(s, dict) else str(s)
        if src and src not in seen:
            seen.add(src)
            sources.append(src)
    src_block = "\n".join(f"  - {s}" for s in sources) if sources else "  (none)"
    return f"{answer}\n\nSources:\n{src_block}"


@mcp.tool()
def recall_add_note(title: str, text: str) -> str:
    """Save a new note into Recall and index it immediately (embeds just this
    note, no full rebuild). Returns the source label of the saved note.
    """
    _load()
    source = ingest.add_note(title, text, _embedder())
    return f"Saved and indexed: {source}"


@mcp.tool()
def recall_reindex() -> str:
    """Bring the index in line with notes on disk: embed new/changed notes, drop
    deleted ones. Use after notes were edited outside recall_add_note.
    """
    _load()
    r = ingest.update_index(embedder=_embedder())
    added = [x["name"] for x in r["added"]]
    changed = [x["name"] for x in r["changed"]]
    removed = [x["name"] for x in r["removed"]]
    if not (added or changed or removed):
        return "Index already up to date. Nothing changed."
    return (
        f"added ({len(added)}): {added}\n"
        f"changed ({len(changed)}): {changed}\n"
        f"removed ({len(removed)}): {removed}"
    )


@mcp.tool()
def recall_areas() -> str:
    """List the area filters a search can be restricted to (defined by AREAS in
    config.py).
    """
    _load()
    return ", ".join(rag.list_areas())


def _warm_background():
    """Preload the modules and models off the critical path."""
    try:
        _load()
        rag.warm()
        _embedder()
    except Exception:
        pass


if __name__ == "__main__":
    # Warm in the BACKGROUND so the server answers the MCP handshake immediately.
    # A tool call that arrives first simply waits for _load() to finish.
    threading.Thread(target=_warm_background, daemon=True).start()
    mcp.run()
