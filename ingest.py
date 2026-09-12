# ingest.py
# THE ONE-TIME SETUP. Run this to read your notes and build the searchable index.
# Run it again whenever your notes change.
#
# What it does, in order:
#   1. Read every .md file in the corpus folders.
#   2. Cut each file into chunks (bite-sized pieces).
#   3. Turn each chunk into meaning-numbers (an embedding).
#   4. Save chunk text + numbers into SQLite.
#
# Run from a terminal:  python ingest.py

import shutil
import sqlite3
import subprocess
from datetime import datetime

import numpy as np
from fastembed import TextEmbedding

import config


def pull_mirrors():
    """Fetch the latest content of each read-only git mirror in config.GIT_MIRRORS
    before re-indexing, so a refresh also picks up teammates' new/edited files.

    Strictly read-only toward the REMOTE: we only ever run `git pull --ff-only`,
    which reads the remote and fast-forwards the local working tree — it never
    pushes, never creates merge commits, and so can never alter the source repo.
    Degrades gracefully: if git is missing, the folder isn't a repo, we're
    offline, or the branch has diverged (ff-only refuses), we just skip that
    mirror and index whatever is already on disk. Returns a list of per-mirror
    status dicts: {"name", "ok", "detail"}.
    """
    results = []
    git = shutil.which("git")
    for path in getattr(config, "GIT_MIRRORS", []):
        name = path.name
        if git is None:
            results.append({"name": name, "ok": False, "detail": "git not found"})
            continue
        if not (path / ".git").exists():
            results.append({"name": name, "ok": False, "detail": "not a git clone"})
            continue
        try:
            proc = subprocess.run(
                [git, "-C", str(path), "pull", "--ff-only"],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode == 0:
                out = (proc.stdout or "").strip().splitlines()
                detail = out[-1] if out else "up to date"
                results.append({"name": name, "ok": True, "detail": detail})
            else:
                # Offline, auth, or diverged — keep the existing on-disk copy.
                err = (proc.stderr or proc.stdout or "pull failed").strip().splitlines()
                results.append({"name": name, "ok": False,
                                "detail": err[-1] if err else "pull failed"})
        except subprocess.TimeoutExpired:
            results.append({"name": name, "ok": False, "detail": "pull timed out"})
        except Exception as exc:  # never let a mirror pull break the refresh
            results.append({"name": name, "ok": False, "detail": str(exc)})
    return results


def _normalise(vector):
    """Scale a vector to length 1 so a dot product equals cosine similarity."""
    v = np.asarray(vector, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _slugify(title):
    """Make a safe filename fragment from a title."""
    s = "".join(ch.lower() if ch.isalnum() else "-" for ch in title).strip("-")
    while "--" in s:
        s = s.replace("--", "-")
    return s or "note"


def add_note(title, text, embedder):
    """Save a note the user typed into the page, and add it to the index right
    away (embed just this note — no full rebuild). Returns its source label.
    """
    folder = config.MY_NOTES_DIR
    folder.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{_slugify(title)}-{stamp}.md"
    body = f"# {title}\n\n{text}\n" if title else f"{text}\n"
    path = folder / filename
    path.write_text(body, encoding="utf-8")

    label = folder.name  # "my_notes"
    source = f"{label}/{filename}"
    conn = _connect()
    _embed_file(conn, source, label, filename, path, path.stat().st_mtime, embedder)
    conn.commit()
    conn.close()
    return source


def chunk_text(text):
    """Cut one note into pieces of roughly CHUNK_CHAR_TARGET characters.

    We split on blank lines first (paragraphs / sections), then greedily pack
    those blocks together until we hit the target size. This keeps related
    sentences in the same chunk instead of slicing mid-thought.
    """
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    chunks = []
    current = ""
    for block in blocks:
        # If adding this block would overflow the target, close the current chunk.
        if current and len(current) + len(block) + 2 > config.CHUNK_CHAR_TARGET:
            chunks.append(current.strip())
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current.strip():
        chunks.append(current.strip())
    return chunks


def readable_title(rel_path):
    """Turn a note's relative path into a plain topic label, so the search has
    topic words to match against. Examples:
      'reference_brew_guide.md' -> 'brew guide'
      'team_maria.md'           -> 'team maria'
    """
    rel = rel_path.replace("\\", "/")
    parts = rel.rsplit(".", 1)[0].split("/")  # drop extension, split folders
    for prefix in ("project_", "feedback_", "reference_", "user_"):
        if parts[-1].startswith(prefix):
            parts[-1] = parts[-1][len(prefix):]
            break
    return " ".join(parts).replace("_", " ")


def corpus_folders():
    """Return (folder, label) for every notes folder to index:
    each project's memory folder under CLAUDE_PROJECTS_ROOT (labelled by project),
    plus each explicit extra folder in CORPUS_FOLDERS (labelled by its own name).
    """
    out = []
    root = config.CLAUDE_PROJECTS_ROOT
    if root.exists():
        for mem in sorted(root.glob("*/memory")):
            if mem.is_dir():
                label = mem.parent.name
                if label.startswith("C--"):
                    label = label[3:]
                out.append((mem, label))
    for f in config.CORPUS_FOLDERS:
        if f.exists():
            out.append((f, f.name))
    return out


def _note_paths(folder):
    """Every .md under a folder (including sub-folders), skipping junk dirs and
    the summary index files."""
    for path in sorted(folder.rglob("*.md")):
        if path.name in config.SKIP_FILES:
            continue
        if any(part in config.SKIP_DIRS for part in path.parts):
            continue
        yield path


def _connect():
    """Open the index database, creating the tables the first time."""
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=30.0)
    conn.execute("PRAGMA busy_timeout=30000")
    # WAL lets a reader and a writer coexist, so the web app and the MCP server
    # can touch this DB at once without "database is locked". It is a file-level
    # mode: set once, it sticks.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks ("
        "id INTEGER PRIMARY KEY, source TEXT, chunk_index INTEGER, text TEXT, embedding BLOB)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
    # Remembers each note's last-modified time so we only re-embed what changed.
    conn.execute("CREATE TABLE IF NOT EXISTS files (source TEXT PRIMARY KEY, mtime REAL)")
    return conn


def current_files():
    """Every note on disk now, as {source: (path, label, rel, mtime)}."""
    out = {}
    for folder, label in corpus_folders():
        for path in _note_paths(folder):
            rel = path.relative_to(folder).as_posix()
            out[f"{label}/{rel}"] = (path, label, rel, path.stat().st_mtime)
    return out


def _embed_file(conn, source, label, rel, path, mtime, embedder):
    """(Re)index one note. Embed FIRST (the slow part) with no DB lock held, then
    do the drop + inserts in one quick write, so a concurrent reader/writer is
    not blocked while the embedding model runs."""
    title = readable_title(rel)
    text = path.read_text(encoding="utf-8", errors="ignore")
    stamped = [f"[{label} — {title}]\n{p}" for p in chunk_text(text)]
    vectors = [_normalise(v) for v in embedder.embed(stamped)] if stamped else []
    conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
    for i, (piece, vec) in enumerate(zip(stamped, vectors)):
        conn.execute(
            "INSERT INTO chunks (source, chunk_index, text, embedding) VALUES (?, ?, ?, ?)",
            (source, i, piece, vec.tobytes()),
        )
    conn.execute("INSERT OR REPLACE INTO files (source, mtime) VALUES (?, ?)", (source, mtime))


def _display_name(source):
    """A short, human-friendly label for a note source ('label/rel/path.md'),
    e.g. 'sample_notes · client brightwater'."""
    label, _, rel = source.partition("/")
    title = readable_title(rel) if rel else source
    return f"{label} · {title}" if label else title


def update_index(embedder=None, verbose=False):
    """Bring the index in line with the notes on disk. Embeds ONLY notes that
    are new or changed, and drops notes that were deleted. Returns a dict:
      {"added": [...], "changed": [...], "removed": [...]}
    each a list of human-friendly note names. Fast when little or nothing changed.
    """
    conn = _connect()
    disk = current_files()
    indexed = dict(conn.execute("SELECT source, mtime FROM files").fetchall())

    added = [s for s in disk if s not in indexed]
    changed = [s for s in disk if s in indexed and disk[s][3] > indexed[s] + 1e-6]
    to_embed = added + changed
    to_delete = [s for s in indexed if s not in disk]

    if not to_embed and not to_delete:
        conn.close()
        return {"added": [], "changed": [], "removed": []}

    if to_embed and embedder is None:
        if verbose:
            print(f"Loading embedding model ({config.EMBED_MODEL})...")
        embedder = TextEmbedding(model_name=config.EMBED_MODEL)

    # Commit incrementally so the write lock is only ever held for a quick
    # delete/insert, never across the slow embedding. This lets the web app and
    # the MCP server share the DB without long "database is locked" waits.
    for source in to_delete:
        conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
        conn.execute("DELETE FROM files WHERE source = ?", (source,))
        if verbose:
            print(f"  removed {source}")
    if to_delete:
        conn.commit()

    for source in to_embed:
        path, label, rel, mtime = disk[source]
        _embed_file(conn, source, label, rel, path, mtime, embedder)  # embeds first (no lock), then writes
        conn.commit()  # release the write lock after each note
        if verbose:
            print(f"  indexed {source}")

    conn.close()
    def items(srcs):
        return [{"name": _display_name(s), "source": s} for s in srcs]
    return {
        "added": items(added),
        "changed": items(changed),
        "removed": items(to_delete),
    }


def build_index(embedder=None):
    """Full rebuild from scratch: forget everything, then index every note."""
    if config.DB_PATH.exists():
        conn = sqlite3.connect(config.DB_PATH, timeout=30.0)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute("DROP TABLE IF EXISTS files")
        conn.commit()
        conn.close()
    print("Indexing all notes (one-time full build)...")
    result = update_index(embedder=embedder, verbose=True)
    print(f"\nDone. Indexed {len(result['added']) + len(result['changed'])} notes into {config.DB_PATH}")


if __name__ == "__main__":
    build_index()
