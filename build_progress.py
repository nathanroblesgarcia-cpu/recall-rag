# build_progress.py
# One-off visible index build. Same logic as ingest.update_index, but it reports
# CHUNK-level progress and commits in small batches, so even a single very large
# file (the ~1,200-chunk Standard Model docs) shows smooth movement and is
# resumable mid-file. Safe to re-run: already-indexed files are skipped, and a
# file only counts as done once fully committed.
import time

from fastembed import TextEmbedding

import config
import ingest

BATCH = 64  # chunks embedded + committed per step (short write-lock windows)


def _retry(fn, tries=6, delay=5.0):
    """Run a DB write, retrying on a transient 'database is locked'. The recall-mcp
    server shares this database and can hold a brief lock; rather than crash the
    whole build, wait and retry so we ride through it."""
    import sqlite3 as _sq
    for attempt in range(tries):
        try:
            return fn()
        except _sq.OperationalError as exc:
            if "locked" in str(exc).lower() and attempt < tries - 1:
                time.sleep(delay)
                continue
            raise


def bar(done, total, label=""):
    width = 30
    frac = done / total if total else 1.0
    filled = int(width * frac)
    b = "█" * filled + "░" * (width - filled)
    pct = int(frac * 100)
    line = f"[{b}] {pct:3d}%  {done}/{total} chunks  {label}"
    print(line[:120], flush=True)


def main():
    embedder = TextEmbedding(model_name=config.EMBED_MODEL)
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
    conn.commit()

    # Pre-count chunks across all pending files so the bar reflects real work,
    # not file count (files range from 1 to 1,200+ chunks).
    plan = []          # (source, label, rel, mtime, [stamped chunks])
    total_chunks = 0
    for source in to_embed:
        path, label, rel, mtime = disk[source]
        title = ingest.readable_title(rel)
        text = path.read_text(encoding="utf-8", errors="ignore")
        stamped = [f"[{label} — {title}]\n{p}" for p in ingest.chunk_text(text)]
        plan.append((source, label, rel, mtime, stamped))
        total_chunks += len(stamped)

    print(f"START files={len(to_embed)} chunks={total_chunks} delete={len(to_delete)}", flush=True)
    if total_chunks == 0:
        print("DONE nothing to embed", flush=True)
        return

    t0 = time.time()
    done = 0
    for source, label, rel, mtime, stamped in plan:
        # Clear any partial prior attempt for this file, then embed + insert its
        # chunks in small batches so progress is visible and the write-lock is
        # only held for each short batch commit.
        _retry(lambda: (conn.execute("DELETE FROM chunks WHERE source = ?", (source,)), conn.commit()))
        for start in range(0, len(stamped), BATCH):
            batch = stamped[start:start + BATCH]
            # parallel=1 keeps embedding in THIS process. Without it fastembed can
            # spawn worker subprocesses on Windows that re-open the database and
            # deadlock against this build's own write-lock.
            vectors = [ingest._normalise(v) for v in embedder.embed(batch, parallel=1)]
            def write_batch():
                for j, (piece, vec) in enumerate(zip(batch, vectors)):
                    conn.execute(
                        "INSERT INTO chunks (source, chunk_index, text, embedding) VALUES (?, ?, ?, ?)",
                        (source, start + j, piece, vec.tobytes()),
                    )
                conn.commit()
            _retry(write_batch)
            done += len(batch)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0
            eta = int((total_chunks - done) / rate) if rate else 0
            bar(done, total_chunks, f"eta {eta // 60}m{eta % 60:02d}s  {rel[-38:]}")
        # Mark the file done only after all its chunks are in — this is what makes
        # an interrupted run resume cleanly (a half-done file is redone, not skipped).
        _retry(lambda: (conn.execute("INSERT OR REPLACE INTO files (source, mtime) VALUES (?, ?)", (source, mtime)), conn.commit()))
    conn.close()
    print(f"DONE {len(to_embed)} files, {total_chunks} chunks in {int(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # guard so any spawned worker can't re-run main()
    main()
