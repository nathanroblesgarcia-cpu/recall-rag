# indexer.py
# BACKGROUND INDEXING. Runs the same incremental re-embed as ingest.update_index,
# but off the web request / startup thread, in small batches, with live progress.
#
# Why this exists: embedding the corpus is CPU work that can take minutes. Doing it
# inline blocked the whole app — startup wouldn't bind the port until it finished,
# and clicking Refresh froze the page. Here the app comes up instantly and Refresh
# returns immediately; the page polls status() and shows a progress bar while this
# thread embeds behind it.
#
# One build at a time (a lock guards start), resumable (a file is marked done only
# after all its chunks commit), and lock-tolerant (retries transient
# 'database is locked' while the recall-mcp server holds a brief lock).

import threading
import time

import config
import ingest

BATCH = 64  # chunks embedded + committed per step (short write-lock windows)

_lock = threading.Lock()

# Shared progress, read by the /refresh_status route. Plain dict: the worker only
# ever replaces whole values and readers only read, so no per-field locking needed.
_state = {
    "running": False,
    "phase": "idle",      # idle | mirrors | scanning | embedding | done | error
    "done": 0,            # chunks embedded so far this run
    "total": 0,           # chunks to embed this run
    "eta": 0,             # seconds remaining, best estimate
    "message": "",
    "delta": None,        # {"added","changed","removed"} when finished
    "mirrors": None,      # mirror pull results when finished
    "error": None,
    "started_at": 0.0,
    "finished_at": 0.0,
}


def status():
    """A snapshot of the current build state, for the page to poll."""
    return dict(_state)


def is_running():
    return _state["running"]


def start(embedder, run_mirrors=True, freeze_journal=True):
    """Kick off a background index if one isn't already running. Returns True if
    this call started a build, False if one was already in flight."""
    if not _lock.acquire(blocking=False):
        return False
    if _state["running"]:
        _lock.release()
        return False
    _state.update(
        running=True, phase="starting", done=0, total=0, eta=0,
        message="Starting…", delta=None, mirrors=None, error=None,
        started_at=time.time(), finished_at=0.0,
    )
    t = threading.Thread(
        target=_run, args=(embedder, run_mirrors, freeze_journal), daemon=True
    )
    t.start()
    return True


def _retry(fn, tries=6, delay=5.0):
    """Run a DB write, retrying transient 'database is locked' (the recall-mcp
    server shares this DB and can hold a brief lock)."""
    import sqlite3 as _sq
    for attempt in range(tries):
        try:
            return fn()
        except _sq.OperationalError as exc:
            if "locked" in str(exc).lower() and attempt < tries - 1:
                time.sleep(delay)
                continue
            raise


def _run(embedder, run_mirrors, freeze_journal):
    try:
        if run_mirrors:
            _state.update(phase="mirrors", message="Fetching shared notes…")
            _state["mirrors"] = ingest.pull_mirrors()

        _state.update(phase="scanning", message="Checking what changed…")
        conn = ingest._connect()
        disk = ingest.current_files()
        indexed = dict(conn.execute("SELECT source, mtime FROM files").fetchall())

        added = [s for s in disk if s not in indexed]
        changed = [s for s in disk if s in indexed and disk[s][3] > indexed[s] + 1e-6]
        to_embed = added + changed
        to_delete = [s for s in indexed if s not in disk]

        for s in to_delete:
            conn.execute("DELETE FROM chunks WHERE source = ?", (s,))
            conn.execute("DELETE FROM files WHERE source = ?", (s,))
        if to_delete:
            conn.commit()

        # Pre-chunk every pending file so the bar reflects real chunk work, not
        # file count (files range from 1 to 1,200+ chunks).
        plan = []          # (source, mtime, [stamped chunks])
        total_chunks = 0
        for source in to_embed:
            path, label, rel, mtime = disk[source]
            title = ingest.readable_title(rel)
            text = path.read_text(encoding="utf-8", errors="ignore")
            stamped = [f"[{label} — {title}]\n{p}" for p in ingest.chunk_text(text)]
            plan.append((source, mtime, stamped))
            total_chunks += len(stamped)

        _state.update(phase="embedding", total=total_chunks, done=0,
                      message=f"Re-reading {len(to_embed)} note(s)…")

        if total_chunks:
            t0 = time.time()
            done = 0
            for source, mtime, stamped in plan:
                _retry(lambda: (conn.execute("DELETE FROM chunks WHERE source = ?", (source,)), conn.commit()))
                for start_i in range(0, len(stamped), BATCH):
                    batch = stamped[start_i:start_i + BATCH]
                    # Default parallelism = onnxruntime's in-process threads, which
                    # is fast AND safe here (this is exactly what update_index used
                    # at ~70 chunks/s). We deliberately do NOT pass parallel=0/>1:
                    # those spawn Windows subprocesses that re-open the DB and can
                    # deadlock against this build's write-lock.
                    vectors = [ingest._normalise(v) for v in embedder.embed(batch)]

                    def write_batch():
                        for j, (piece, vec) in enumerate(zip(batch, vectors)):
                            conn.execute(
                                "INSERT INTO chunks (source, chunk_index, text, embedding) VALUES (?, ?, ?, ?)",
                                (source, start_i + j, piece, vec.tobytes()),
                            )
                        conn.commit()
                    _retry(write_batch)

                    done += len(batch)
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed else 0
                    _state.update(
                        done=done,
                        eta=int((total_chunks - done) / rate) if rate else 0,
                    )
                # Mark the file done only once all its chunks are in, so an
                # interrupted run redoes a half-done file rather than skipping it.
                _retry(lambda: (conn.execute("INSERT OR REPLACE INTO files (source, mtime) VALUES (?, ?)", (source, mtime)), conn.commit()))

        conn.close()

        if freeze_journal:
            try:
                import journal
                journal.freeze_past_days()
            except Exception:
                pass  # journal freezing must never fail a build

        def names(srcs):
            return [{"name": ingest._display_name(s), "source": s} for s in srcs]
        _state["delta"] = {
            "added": names(added), "changed": names(changed), "removed": names(to_delete),
        }
        _state.update(phase="done", message="Done.", eta=0, finished_at=time.time())
    except Exception as exc:
        _state.update(phase="error", error=str(exc), message=f"Indexing failed: {exc}",
                      finished_at=time.time())
    finally:
        _state["running"] = False
        if _lock.locked():
            _lock.release()
