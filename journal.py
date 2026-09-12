# journal.py
# A TRUE day-by-day journal. Each day becomes its own entry that is FROZEN once
# the day is over, so it never changes again even if you edit those notes later.
#
# How it works:
#   - Every note in the index has a "last changed" time (the files table).
#   - For any day that is already over, we snapshot the notes changed that day
#     into journal_activity and mark the day frozen. Frozen days are read back
#     from that snapshot, not recomputed, so history stays put.
#   - Today is shown live (it is still in progress) until it, too, gets frozen
#     the next time the app runs on a later day.
#   - You can also write your own words for any day (journal_note); those are
#     always kept.

import datetime
import sqlite3
from collections import defaultdict

import config
import rag
from rag import _map_label


def _label(source):
    """Readable title for a note, from its path (same as the map uses)."""
    return _map_label(source.partition("/")[2] or source)


def _connect():
    conn = sqlite3.connect(config.DB_PATH, timeout=5.0)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS journal_activity "
        "(date TEXT, source TEXT, label TEXT, PRIMARY KEY (date, source))"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS journal_frozen (date TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE IF NOT EXISTS journal_note (date TEXT PRIMARY KEY, note TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS journal_summary (date TEXT PRIMARY KEY, summary TEXT)")
    return conn


def _day_of(mtime):
    return datetime.date.fromtimestamp(mtime).isoformat()


def freeze_past_days():
    """Snapshot every finished day (before today) that has not been frozen yet.
    Safe to call often: already-frozen days are skipped.

    Never fatal: if the database is momentarily write-locked (e.g. a background
    index build is committing), skip this pass quietly. Freezing is idempotent,
    so the next call catches up — better than 500-ing a page load."""
    try:
        conn = _connect()
        today = datetime.date.today().isoformat()
        frozen = {d for (d,) in conn.execute("SELECT date FROM journal_frozen").fetchall()}

        by_day = defaultdict(list)
        for source, mtime in conn.execute("SELECT source, mtime FROM files").fetchall():
            by_day[_day_of(mtime)].append(source)

        for day, sources in by_day.items():
            if day >= today or day in frozen:   # never freeze today or the future
                continue
            for s in sources:
                conn.execute(
                    "INSERT OR IGNORE INTO journal_activity (date, source, label) VALUES (?, ?, ?)",
                    (day, s, _label(s)),
                )
            conn.execute("INSERT OR IGNORE INTO journal_frozen (date) VALUES (?)", (day,))

        conn.commit()
        conn.close()
    except sqlite3.OperationalError:
        return  # database busy (build committing) — catch up on the next call


def _today_activity(conn, today_iso):
    """Notes changed today, computed live (today is not frozen yet)."""
    out = []
    for source, mtime in conn.execute("SELECT source, mtime FROM files").fetchall():
        if _day_of(mtime) == today_iso:
            out.append({"source": source, "label": _label(source)})
    return sorted(out, key=lambda a: a["label"].lower())


def _pretty_date(iso):
    d = datetime.date.fromisoformat(iso)
    return d.strftime("%A %d %B %Y")


def set_note(date, note):
    """Save (or clear) the user's own words for a day."""
    conn = _connect()
    note = (note or "").strip()
    if note:
        conn.execute(
            "INSERT INTO journal_note (date, note) VALUES (?, ?) "
            "ON CONFLICT(date) DO UPDATE SET note = excluded.note",
            (date, note),
        )
    else:
        conn.execute("DELETE FROM journal_note WHERE date = ?", (date,))
    conn.commit()
    conn.close()


def _day_sources(conn, date, today_iso):
    """The note sources for a day: live if today, frozen snapshot otherwise."""
    if date == today_iso:
        return [a["source"] for a in _today_activity(conn, today_iso)]
    return [s for (s,) in conn.execute(
        "SELECT source FROM journal_activity WHERE date = ? ORDER BY label", (date,)
    ).fetchall()]


def make_summary(date):
    """Write (or rewrite) the one-line summary for a day with the local model,
    store it, and return it. Returns '' if there is nothing to summarise."""
    conn = _connect()
    today_iso = datetime.date.today().isoformat()
    sources = _day_sources(conn, date, today_iso)
    if not sources:
        conn.close()
        return ""
    label = "today" if date == today_iso else _pretty_date(date)
    summary = (rag.summarise_notes(sources, label) or "").strip()
    if summary:
        conn.execute(
            "INSERT INTO journal_summary (date, summary) VALUES (?, ?) "
            "ON CONFLICT(date) DO UPDATE SET summary = excluded.summary",
            (date, summary),
        )
        conn.commit()
    conn.close()
    return summary


def journal_days():
    """Every day that has activity or a written note, newest first. Today is
    always shown. Frozen days come from the snapshot; today is live."""
    freeze_past_days()
    conn = _connect()
    today_iso = datetime.date.today().isoformat()

    notes = dict(conn.execute("SELECT date, note FROM journal_note").fetchall())
    summaries = dict(conn.execute("SELECT date, summary FROM journal_summary").fetchall())
    live = _today_activity(conn, today_iso)

    dates = set()
    dates.update(d for (d,) in conn.execute("SELECT DISTINCT date FROM journal_activity").fetchall())
    dates.update(notes.keys())
    dates.add(today_iso)  # always show today, even if nothing changed yet

    out = []
    for d in sorted(dates, reverse=True):
        if d == today_iso:
            activity = live
        else:
            activity = [
                {"source": s, "label": lab}
                for s, lab in conn.execute(
                    "SELECT source, label FROM journal_activity WHERE date = ? ORDER BY label",
                    (d,),
                ).fetchall()
            ]
        out.append({
            "date": d,
            "pretty": _pretty_date(d),
            "is_today": d == today_iso,
            "activity": activity,
            "note": notes.get(d, ""),
            "summary": summaries.get(d, ""),
        })
    conn.close()
    return out
