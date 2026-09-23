# rag.py
# THE CORE. This is the R and the G of RAG.
#   Retrieve: turn a question into meaning-numbers, find the closest chunks.
#   Generate: hand those chunks to Claude and ask it to answer from them.
#
# app.py calls answer_question() for every question the user types.

import datetime
import os
import re
import sqlite3
import numpy as np
from fastembed import TextEmbedding
from dotenv import load_dotenv

import config

load_dotenv()  # reads your .env so ANTHROPIC_API_KEY becomes available


def _db():
    """Open the index database for reading. busy_timeout makes a query WAIT up to
    5s for a lock to clear (e.g. while a background index build commits a file)
    instead of failing instantly with 'database is locked', so the app stays
    usable while a build is running."""
    conn = sqlite3.connect(config.DB_PATH, timeout=5.0)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

# Load the embedding model once when this file is first imported, not on every
# question. Loading it is slow; using it is fast. We reuse the same one.
_embedder = TextEmbedding(model_name=config.EMBED_MODEL)

# A system-level rule the small model follows more reliably than the same words
# buried in the user prompt: answer strictly from the supplied notes, and admit
# when they don't cover it instead of guessing.
_SYSTEM = (
    f"You are the personal memory of {config.USER_NAME}. You are talking directly "
    f"to {config.USER_FIRST_NAME}. When they say 'I', 'me', 'my', 'mine', or "
    f"'myself', that means {config.USER_FIRST_NAME}, and notes about "
    f"{config.USER_FIRST_NAME} (their money, work, plans, life) are about the "
    "person asking. Answer questions strictly from their own notes, which are "
    "supplied with each question. Use only what the notes say. If the notes do "
    "not contain the answer, say so plainly rather than guessing. Be tight and "
    "factual, name the source file(s) you drew from, and do not invent details."
)


# Manual per-note area overrides live in a small table in index.db and take
# precedence over the rules, so the user can reclassify any note from the page
# and have it stick. Cached in memory; reloaded whenever an override is saved.
_AREA_OVERRIDES = None


def _ensure_area_table(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS note_area (source TEXT PRIMARY KEY, area TEXT)")


def _area_overrides():
    """The saved {source: area} overrides, loaded once and cached."""
    global _AREA_OVERRIDES
    if _AREA_OVERRIDES is None:
        conn = _db()
        _ensure_area_table(conn)
        _AREA_OVERRIDES = {r[0]: r[1] for r in conn.execute("SELECT source, area FROM note_area")}
        conn.close()
    return _AREA_OVERRIDES


def _rule_area(source):
    """The area the RULES alone would assign (ignoring any manual override)."""
    s = (source or "").lower()
    for name, patterns in config.AREAS:
        if any(p in s for p in patterns):
            return name
    return config.AREAS[-1][0]


def area_of(source):
    """Which life area a note belongs to. A manual override wins; otherwise the
    rules decide (first matching area, catch-all last)."""
    ov = _area_overrides()
    if source in ov:
        return ov[source]
    return _rule_area(source)


def set_area_override(source, area):
    """Manually file a note into an area. Saving the area the rules would have
    picked anyway just clears the override (keeps the table tidy). Returns the
    stored area, or None if the area name is unknown."""
    if area not in {name for name, _ in config.AREAS}:
        return None
    conn = _db()
    _ensure_area_table(conn)
    if area == _rule_area(source):
        conn.execute("DELETE FROM note_area WHERE source = ?", (source,))
    else:
        conn.execute(
            "INSERT INTO note_area (source, area) VALUES (?, ?) "
            "ON CONFLICT(source) DO UPDATE SET area = excluded.area",
            (source, area),
        )
    conn.commit()
    conn.close()
    global _AREA_OVERRIDES
    _AREA_OVERRIDES = None  # force reload on next read
    return area


def list_areas():
    """The area names in order, for the page's 'Ask about…' picker."""
    return [name for name, _ in config.AREAS]


def categorised_notes():
    """Every note grouped by the life area it falls into, with friendly names, so
    the page can show the full categorisation after a refresh and the user can
    spot anything filed in the wrong area. Returns an ordered list of
    {"area", "notes": [{"name", "source"}]} — areas in config order, notes A→Z."""
    import ingest
    conn = _db()
    sources = sorted({r[0] for r in conn.execute("SELECT source FROM chunks")})
    conn.close()
    overrides = _area_overrides()
    by_area = {name: [] for name, _ in config.AREAS}
    for s in sources:
        by_area[area_of(s)].append({
            "name": ingest._display_name(s),
            "source": s,
            "overridden": s in overrides,
        })
    return [{"area": name, "notes": by_area[name]} for name, _ in config.AREAS]


# In-memory copy of every chunk + vector (v1.16 speed fix). Re-reading every
# chunk from disk on each search gets slow as a note set grows. Now it is read once and reused
# until the index changes. Stored as (signature, chunks, matrix).
_CHUNK_CACHE = None


def _index_signature(conn):
    """A cheap fingerprint of the chunks table: (row count, highest id). Any note
    added, changed (deleted + re-inserted with new ids) or removed moves one of
    them. Works across processes, so a note saved by the MCP server or a refresh
    in the web app is picked up on the very next search."""
    return conn.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks").fetchone()


def _load_all_chunks():
    """Every chunk and its vector, from the in-memory cache when the index has
    not changed, otherwise freshly read from SQLite.

    Callers must treat the returned list and matrix as read-only (retrieve
    builds new lists/copies, never edits them in place).

    When your corpus grows much bigger, this is the moment you switch to
    pgvector (Phase 2 of the plan) so the database does the search instead of
    Python. Not yet.
    """
    global _CHUNK_CACHE
    conn = _db()
    sig = _index_signature(conn)
    if _CHUNK_CACHE is not None and _CHUNK_CACHE[0] == sig:
        conn.close()
        return _CHUNK_CACHE[1], _CHUNK_CACHE[2]
    rows = conn.execute(
        "SELECT id, source, chunk_index, text, embedding FROM chunks"
    ).fetchall()
    conn.close()

    chunks = []
    vectors = []
    for cid, source, idx, text, blob in rows:
        chunks.append({"id": cid, "source": source, "chunk_index": idx, "text": text})
        vectors.append(np.frombuffer(blob, dtype=np.float32))
    matrix = np.vstack(vectors) if vectors else np.empty((0, config.EMBED_DIMS))
    _CHUNK_CACHE = (sig, chunks, matrix)
    return chunks, matrix


def _day_bounds(d):
    """(start, end) unix timestamps covering the whole calendar day `d`."""
    start = datetime.datetime.combine(d, datetime.time.min).timestamp()
    end = datetime.datetime.combine(d, datetime.time.max).timestamp()
    return start, end


def _time_window(question):
    """If the question is about a time ('yesterday', 'today', 'this week',
    'recently'...), return (label, start_ts, end_ts). Otherwise None. This is
    what lets Recall answer 'what did I do yesterday' from WHEN notes changed,
    instead of trying (and failing) to match those words by meaning."""
    q = question.lower()
    today = datetime.date.today()
    if "yesterday" in q:
        d = today - datetime.timedelta(days=1)
        s, _ = _day_bounds(d); _, e = _day_bounds(d)
        return ("yesterday", s, e)
    if re.search(r"\btoday\b|this morning|so far today|earlier today", q):
        s, e = _day_bounds(today)
        return ("today", s, e)
    if "this week" in q:
        start = today - datetime.timedelta(days=today.weekday())
        s, _ = _day_bounds(start); _, e = _day_bounds(today)
        return ("this week", s, e)
    if "last week" in q:
        this_mon = today - datetime.timedelta(days=today.weekday())
        s, _ = _day_bounds(this_mon - datetime.timedelta(days=7))
        _, e = _day_bounds(this_mon - datetime.timedelta(days=1))
        return ("last week", s, e)
    if re.search(r"recently|lately|past few days|last few days|past couple", q):
        s, _ = _day_bounds(today - datetime.timedelta(days=7))
        _, e = _day_bounds(today)
        return ("the past week", s, e)
    return None


def _recent_sources(start_ts, end_ts):
    """The notes whose last-changed time falls in the window, newest first."""
    conn = _db()
    rows = conn.execute(
        "SELECT source, mtime FROM files WHERE mtime >= ? AND mtime <= ? ORDER BY mtime DESC",
        (start_ts, end_ts),
    ).fetchall()
    conn.close()
    return rows


def _retrieve_recent(window, limit=12):
    """Build retrieval results from a time window: the notes changed then, as
    'sources' the answer step can summarise. Score is a recency rank (newest
    highest) so the sources list still sorts sensibly."""
    label, start_ts, end_ts = window
    recents = _recent_sources(start_ts, end_ts)
    if not recents:
        return [{"source": f"(no activity {label})", "chunk_index": 0,
                 "text": f"No notes were changed {label}.", "score": 1.0, "recent": True}]
    results = []
    n = min(len(recents), limit)
    for i, (source, mtime) in enumerate(recents[:limit]):
        when = datetime.datetime.fromtimestamp(mtime).strftime("%a %d %b %H:%M")
        body = note_text(source)
        results.append({
            "source": source, "chunk_index": 0,
            "text": f"[last changed {when}]\n{body[: config.GEN_CHUNK_CHARS * 2]}",
            "score": round(1.0 - i / max(1, n), 3), "recent": True,
        })
    return results


# Words/phrases that mean "this question refers back to the last one" — a real
# follow-up like "what about the other one?" or "why?". Only when one of these is
# present (or the question is a tiny fragment) do we fold the previous question
# into the search. Otherwise a fresh topic ("how much did I spend last month?"
# asked right after a question about a person) would drag the old topic's notes to the top
# and bury its own answer. Word-boundary matched on the lowercased question.
_FOLLOWUP_MARKERS = re.compile(
    r"\b(it|its|it's|that|this|these|those|them|they|their|there|"
    r"he|she|him|her|his|hers|one|ones|same|above|previous|"
    r"what about|how about|tell me more|more about|and what|and how|"
    r"elaborate|explain that|the other|the first|the second|the last)\b",
    re.IGNORECASE,
)


def _is_followup(question):
    """True when the question reads as a follow-up to the previous turn, so its
    search should borrow the previous question's context. A self-contained new
    question returns False and searches on its own words only."""
    q = (question or "").strip()
    if not q:
        return False
    # First-person questions ("who am I?", "what's my plan?") are about the user,
    # not a follow-up to the last turn — don't fold the previous question in.
    if re.search(r"\b(i|me|my|mine|myself)\b", q, re.IGNORECASE):
        return False
    # A very short fragment ("why?", "and the dividends?") is almost always a
    # follow-up; anything longer must actually contain a referential marker.
    if len(q.split()) <= 3:
        return True
    return bool(_FOLLOWUP_MARKERS.search(q))


def retrieve(question, top_k=None, history=None, area=None):
    """Find the chunks whose meaning is closest to the question.

    Time questions ('what did I do yesterday') are answered differently: by
    which notes changed in that window, not by meaning-search.

    `area` (a life-area name) limits the search to notes in that area only, so a
    Money question never sees Work notes. None/"Everything" searches all notes."""
    window = _time_window(question)
    if window:
        return _retrieve_recent(window)

    top_k = top_k or config.TOP_K
    chunks, matrix = _load_all_chunks()
    if not chunks:
        return []

    # Narrow to one life area if asked. Keep only the chunks whose note belongs
    # to that area; if that leaves nothing, fall back to searching everything so
    # the user always gets an answer rather than a blank.
    if area and area != "Everything":
        keep = [i for i, c in enumerate(chunks) if area_of(c["source"]) == area]
        if keep:
            chunks = [chunks[i] for i in keep]
            matrix = matrix[keep]

    # For a follow-up, fold in the previous question so a vague follow-up like
    # "what about the other one?" still finds the right notes. But only when the
    # question actually reads as a follow-up — a fresh topic searches on its own
    # words, so the previous turn can't drag its notes to the top.
    search_text = question
    if history and _is_followup(question):
        search_text = history[-1]["question"] + " " + question

    # Turn the search text into the same kind of meaning-numbers, and normalise it.
    q = np.asarray(list(_embedder.embed([search_text]))[0], dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)

    # Because every vector is length 1, this dot product IS the cosine similarity:
    # a score from -1 (opposite) to 1 (identical meaning), one score per chunk.
    scores = matrix @ q

    # Rank every chunk by meaning, best first.
    order = np.argsort(scores)[::-1]

    # With reranking on, gather a wider shortlist first, then let the reranker
    # pick the best top_k from it.
    n = max(top_k, config.RERANK_POOL) if config.RERANK else top_k
    if not config.HYBRID_SEARCH:
        top_idx = [int(i) for i in order[:n]]
        found_by = {i: "meaning" for i in top_idx}
    else:
        top_idx, found_by = _fuse(order, _keyword_search(search_text), chunks, n)

    if config.RERANK:
        top_idx = _rerank(search_text, top_idx, chunks)[:top_k]

    results = []
    for i in top_idx:
        item = dict(chunks[i])
        # The score shown stays the meaning score (0 to 1), so the page reads the
        # same; the ORDER is what hybrid changes.
        item["score"] = float(scores[i])
        item["found_by"] = found_by[int(i)]
        results.append(item)
    return results


_RERANKER = None


def _rerank(question, idx, chunks):
    """RERANKING (v1.17): a second, closer read of the shortlist.

    The first searches score the question and each chunk SEPARATELY (meaning
    numbers, word counts) - fast, but rough. A reranker (a cross-encoder) reads
    the question and one chunk TOGETHER and scores how well that chunk answers
    it. Too slow to run on every chunk, fine on a shortlist of ~20. Loaded
    on first use only, so the app starts as fast as before."""
    global _RERANKER
    if not idx:
        return idx
    try:
        if _RERANKER is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _RERANKER = TextCrossEncoder(model_name=config.RERANK_MODEL)
        texts = [chunks[i]["text"] for i in idx]
        scores = list(_RERANKER.rerank(question, texts))
    except Exception as e:
        print(f"[rerank] skipped: {e}")  # never let reranking break an answer
        return idx
    return [i for _, i in sorted(zip(scores, idx), key=lambda p: p[0], reverse=True)]


# Small words that carry no search value on their own. Word-search drops them so
# "What is the port for BeanCount?" searches for just 'port' and 'beancount'.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am", "do", "does",
    "did", "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "i", "me", "my", "mine", "we", "our", "you", "your", "it", "its", "this", "that",
    "these", "those", "of", "in", "on", "at", "to", "for", "from", "by", "with",
    "and", "or", "not", "no", "so", "if", "about", "into", "than", "then", "there",
    "can", "could", "should", "would", "will", "has", "have", "had", "much", "many",
    "any", "some", "tell", "please", "use", "uses", "used", "run", "runs", "s",
}

_FTS_READY = False


def _keyword_search(text, limit=None):
    """WORD-SEARCH half of hybrid: SQLite FTS5 over the chunk text, ranked by
    BM25 (rare words that appear often in a chunk score highest). Returns chunk
    ids, best first. Any word may match (OR), so one exact name like 'Verde' is
    enough to pull a chunk in."""
    global _FTS_READY
    limit = limit or config.HYBRID_POOL
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower())
             if w not in _STOPWORDS and (len(w) > 1 or w.isdigit())]
    if not words:
        return []
    # Quote each word so FTS treats it as plain text, never as query syntax.
    match = " OR ".join(f'"{w}"' for w in dict.fromkeys(words))
    conn = _db()
    try:
        if not _FTS_READY:
            import ingest
            ingest.ensure_fts(conn)  # first run builds the word index from existing chunks
            _FTS_READY = True
        rows = conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"[keyword_search] skipped: {e}")  # never let word-search break an answer
        rows = []
    finally:
        conn.close()
    return [r[0] for r in rows]


def _fuse(meaning_order, keyword_ids, chunks, top_k):
    """Merge the two ranked lists with Reciprocal Rank Fusion (RRF).

    Each list gives a chunk 1 / (RRF_K + rank). A chunk near the top of BOTH
    lists adds up two big shares and wins; a chunk only one list liked still
    counts. RRF only looks at ranks, so the two searches' very different score
    scales never need to be compared. Returns (chunk positions, found_by)."""
    k = config.RRF_K
    pool = config.HYBRID_POOL
    fused, found = {}, {}
    for rank, i in enumerate(meaning_order[:pool], start=1):
        i = int(i)
        fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank)
        found[i] = "meaning"
    # Map chunk ids to positions in `chunks` (the area filter may have narrowed
    # it, so a keyword hit outside the chosen area is simply skipped).
    pos = {c["id"]: n for n, c in enumerate(chunks)}
    for rank, cid in enumerate(keyword_ids, start=1):
        i = pos.get(cid)
        if i is None:
            continue
        fused[i] = fused.get(i, 0.0) + config.KEYWORD_WEIGHT / (k + rank)
        found[i] = "both" if i in found else "words"
    best = sorted(fused, key=fused.get, reverse=True)[:top_k]
    return best, found


def _build_prompt(question, retrieved, history=None):
    """Assemble the retrieved chunks and question into one instruction.
    Uses only the top few chunks, trimmed, so the model has less to read before
    it starts answering (faster first word). Time questions get all the recent
    notes and a summarise-my-activity instruction instead."""
    window = _time_window(question)
    today = datetime.date.today().strftime("%A %d %B %Y")

    # For a time question, feed every note in the window (not just the top 3),
    # each trimmed a bit longer, so the summary is complete.
    limit = len(retrieved) if window else config.GEN_CHUNKS
    span = config.GEN_CHUNK_CHARS * 2 if window else config.GEN_CHUNK_CHARS
    used = retrieved[:limit]
    sources = "\n\n".join(
        f"[Source: {r['source']}]\n{r['text'][:span]}" for r in used
    )
    convo = ""
    if history and not window and _is_followup(question):
        prev = history[-1]
        convo = (
            f"EARLIER QUESTION: {prev['question']}\n"
            f"EARLIER ANSWER: {prev.get('answer') or '(no written answer)'}\n\n"
        )

    if window:
        instruction = (
            f"Today is {today}. The notes below are the ones the user changed "
            f"{window[0]}, which is the record of what they worked on {window[0]}. "
            "Summarise what the user worked on, grouped by topic, in plain simple "
            "language, newest first. Be brief. If the only note says there was no "
            "activity, say you have no record of anything changing then."
        )
    else:
        instruction = (
            f"Today is {today}. You are answering {config.USER_FIRST_NAME}'s "
            f"question using only the notes below. 'I', 'me', 'my' mean "
            f"{config.USER_FIRST_NAME}; notes about {config.USER_FIRST_NAME} are "
            "about the person asking. "
            "If this is a follow-up, use the earlier question and answer for context "
            "(for example to resolve what 'the other one' or 'it' refers to). "
            "If the notes do not contain the answer, say so plainly. "
            "Keep the answer tight and factual, and name the source file(s) you used."
        )
    return (
        f"{instruction}\n\n"
        f"{convo}"
        f"NOTES:\n{sources}\n\n"
        f"QUESTION: {question}"
    )


def list_models():
    """Ask Ollama which models are installed locally, so the page can offer a
    picker. Returns (models, current) where each model is
    {"name", "params", "gb"} and current = configured default. `params` is the
    parameter size string like "7.6B" (or None). Empty list if the backend isn't
    ollama or Ollama is unreachable."""
    if config.GEN_BACKEND != "ollama":
        return ([], config.OLLAMA_MODEL)
    try:
        import requests
        base = config.OLLAMA_URL.rsplit("/api/", 1)[0]  # strip "/api/chat"
        resp = requests.get(f"{base}/api/tags", timeout=10)
        resp.raise_for_status()
        models = []
        for m in resp.json().get("models", []):
            details = m.get("details") or {}
            models.append({
                "name": m["name"],
                "params": details.get("parameter_size"),
                "gb": round(m.get("size", 0) / 1e9, 2),
            })
        models.sort(key=lambda x: x["name"])
        return (models, config.OLLAMA_MODEL)
    except Exception as e:
        print(f"[list_models] could not reach Ollama: {e}")
        return ([], config.OLLAMA_MODEL)


def _generate_ollama(prompt, model=None):
    """Send the prompt to the local Ollama server. Free, private, offline."""
    import requests

    resp = requests.post(
        config.OLLAMA_URL,
        json={
            "model": model or config.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {
                "num_ctx": config.OLLAMA_NUM_CTX,
                "temperature": config.OLLAMA_TEMPERATURE,
            },
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def warm(model=None):
    """Load the local model into memory (and keep it there) so the first real
    question doesn't pay the load cost. Safe to call in the background; ignores
    errors if Ollama isn't running yet."""
    if config.GEN_BACKEND != "ollama":
        return
    try:
        import requests
        requests.post(
            config.OLLAMA_URL,
            json={
                "model": model or config.OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "options": {"num_ctx": config.OLLAMA_NUM_CTX, "num_predict": 1},
            },
            timeout=120,
        )
    except Exception as e:
        print(f"[warm] could not preload model: {e}")


def _generate_anthropic(prompt):
    """Send the prompt to Claude in the cloud. Needs ANTHROPIC_API_KEY."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    from anthropic import Anthropic

    client = Anthropic()
    resp = client.messages.create(
        model=config.GEN_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def _stream_ollama(prompt, model=None):
    """Yield the answer from the local Ollama server piece by piece as it writes."""
    import json
    import requests

    with requests.post(
        config.OLLAMA_URL,
        json={
            "model": model or config.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {
                "num_ctx": config.OLLAMA_NUM_CTX,
                "temperature": config.OLLAMA_TEMPERATURE,
            },
        },
        stream=True,
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            chunk = (data.get("message") or {}).get("content", "")
            if chunk:
                yield chunk
            if data.get("done"):
                break


def generate_stream(question, retrieved, history=None, model=None):
    """Yield the answer in small pieces, so the page can show it as it writes.
    Yields nothing when there is no engine or no notes."""
    if config.GEN_BACKEND == "none" or not retrieved:
        return
    prompt = _build_prompt(question, retrieved, history)
    try:
        if config.GEN_BACKEND == "ollama":
            yield from _stream_ollama(prompt, model=model)
        elif config.GEN_BACKEND == "anthropic":
            text = _generate_anthropic(prompt)  # cloud path returns all at once
            if text:
                yield text
    except Exception as e:
        print(f"[generate_stream] {config.GEN_BACKEND} unavailable: {e}")
        yield f"\n\n[The answer engine was unavailable. Check that Ollama is running.]"


def generate(question, retrieved, history=None, model=None):
    """Write the answer from the retrieved chunks, using the configured engine.

    Returns None (so the app still shows retrieval alone) if the engine is set to
    "none", or if the chosen engine is unreachable.
    """
    if config.GEN_BACKEND == "none" or not retrieved:
        return None

    prompt = _build_prompt(question, retrieved, history)
    try:
        if config.GEN_BACKEND == "ollama":
            return _generate_ollama(prompt, model=model)
        if config.GEN_BACKEND == "anthropic":
            return _generate_anthropic(prompt)
    except Exception as e:
        # Don't crash the page if the engine is down; show retrieval and log why.
        print(f"[generate] {config.GEN_BACKEND} unavailable: {e}")
        return None
    return None


def answer_question(question, history=None, model=None, area=None):
    """The whole pipeline in one call: retrieve, then generate.

    Pass `history` (a list of {"question", "answer"} turns) to enable follow-ups.
    Pass `model` to override which local model writes the answer.
    Pass `area` (a life-area name) to search only that slice of the brain.
    """
    retrieved = retrieve(question, history=history, area=area)
    answer = generate(question, retrieved, history=history, model=model)
    return {"question": question, "answer": answer, "sources": retrieved}


def summarise_notes(sources, label="that day"):
    """One short plain-language summary of what a set of notes is about — used by
    the journal's 'Summarise this day' button. Returns None if there is no engine
    or nothing to summarise."""
    if config.GEN_BACKEND == "none" or not sources:
        return None
    # Budget the prompt so a busy day (many notes) stays small and fast. A short
    # slice of each note is plenty to say what the day was about, and it keeps the
    # prompt well under num_ctx so prefill doesn't crawl on a CPU-only machine.
    # Without this, a 16-note day built a ~5.5k-token prompt that timed out.
    used = sources[: config.SUMMARY_MAX_NOTES]
    per_note = max(200, config.SUMMARY_CHAR_BUDGET // max(1, len(used)))
    body = "\n\n".join(
        f"[Source: {s}]\n{note_text(s)[:per_note]}" for s in used
    )
    more = len(sources) - len(used)
    if more > 0:
        body += f"\n\n(and {more} more note{'s' if more != 1 else ''} that day)"
    today = datetime.date.today().strftime("%A %d %B %Y")
    prompt = (
        f"Today is {today}. Below are the notes the user changed on {label}. "
        "In 1 to 3 short sentences, in plain simple language, say what the user "
        "worked on. Group related notes together instead of listing every file. "
        "No preamble and no bullet points, just the summary.\n\n"
        f"NOTES:\n{body}"
    )
    try:
        if config.GEN_BACKEND == "ollama":
            return _generate_ollama(prompt)
        if config.GEN_BACKEND == "anthropic":
            return _generate_anthropic(prompt)
    except Exception as e:
        print(f"[summarise_notes] {config.GEN_BACKEND} unavailable: {e}")
        return None
    return None


def note_text(source):
    """Reconstruct a note's readable text from its stored chunks, so the map can
    show the full note when a dot is clicked. Strips the topic stamp we prepended
    to each chunk."""
    conn = _db()
    rows = conn.execute(
        "SELECT text FROM chunks WHERE source = ? ORDER BY chunk_index", (source,)
    ).fetchall()
    conn.close()
    parts = []
    for (t,) in rows:
        if t.startswith("[") and "\n" in t:  # drop the "[group — title]" stamp line
            t = t.split("\n", 1)[1]
        parts.append(t.strip())
    return "\n\n".join(parts)


def _map_label(rel):
    """Short readable label for a note on the map. Strips filename prefixes and
    date/time junk (from timestamped note files)."""
    rel = rel.replace("\\", "/")
    parts = rel.rsplit(".", 1)[0].split("/")
    for prefix in ("project_", "feedback_", "reference_", "user_"):
        if parts[-1].startswith(prefix):
            parts[-1] = parts[-1][len(prefix):]
            break
    text = " ".join(parts).replace("_", " ").replace("-", " ")
    toks = [t for t in text.split() if not (t.isdigit() and len(t) >= 6)]  # drop 20260819 / 173331
    while toks and toks[0].isdigit():  # drop a leading "03"-style number
        toks.pop(0)
    return " ".join(toks) if toks else text


_STOPWORDS = set(
    "the a an of to and or in on for with by is are was note notes md fix update "
    "summary jun jul aug new all app data his her their via using use plan build "
    "project feedback reference user after before into onto per vs from copy results "
    "block custom chosen this that these those it its as at be do so we you your our "
    "more most less than then setup reduction "
    "sam priya marcus".split()
)


def _kmeans(mat, k, iters=25, seed=42):
    """Group note vectors into k topics by meaning (cosine k-means, numpy only).
    Deterministic (fixed seed) so the same notes always get the same topics."""
    rng = np.random.RandomState(seed)
    n = mat.shape[0]
    k = max(1, min(k, n))
    # k-means++ style seeding
    idx = [rng.randint(n)]
    for _ in range(1, k):
        d = 1 - (mat @ mat[idx].T).max(axis=1)
        d = np.clip(d, 1e-9, None)
        idx.append(int(rng.choice(n, p=d / d.sum())))
    cent = mat[idx].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        assign = (mat @ cent.T).argmax(axis=1)
        for c in range(k):
            members = mat[assign == c]
            if len(members):
                v = members.mean(axis=0)
                cent[c] = v / (np.linalg.norm(v) + 1e-8)
            else:
                cent[c] = mat[rng.randint(n)]
    return assign, k


def _topic_names(labels, assign, k):
    """Name each topic from the most common meaningful words in its notes."""
    names = {}
    for c in range(k):
        counts = {}
        for i, lab in enumerate(labels):
            if assign[i] != c:
                continue
            for w in lab.lower().replace("-", " ").split():
                if len(w) < 3 or w in _STOPWORDS or any(ch.isdigit() for ch in w):
                    continue
                counts[w] = counts.get(w, 0) + 1
        top = sorted(counts, key=lambda w: (-counts[w], w))[:2]
        names[c] = " ".join(_pretty(w) for w in top) if top else f"Topic {c + 1}"
    # keep names unique
    used = {}
    for c in range(k):
        base, nm, i = names[c], names[c], 2
        while nm in used:
            nm = f"{base} {i}"; i += 1
        used[nm] = True; names[c] = nm
    return names


_ACRONYMS = {
    "sql", "api", "csv", "ai", "ml", "rag", "cli", "url", "id", "pos",
}


def _pretty(word):
    lw = word.lower()
    return word.upper() if lw in _ACRONYMS else word.capitalize()


def _core_names(labels, matrix, assign, k):
    """Name each topic core after its most representative note (the one closest
    to the cluster's centre), cleaned up. Reads far better than stitching the
    most common words together."""
    names = {}
    for c in range(k):
        members = [i for i in range(len(labels)) if assign[i] == c]
        if not members:
            names[c] = f"Topic {c + 1}"
            continue
        centre = matrix[members].mean(axis=0)
        centre = centre / (np.linalg.norm(centre) + 1e-8)
        best = max(members, key=lambda i: float(matrix[i] @ centre))
        words = [w for w in labels[best].split() if w.lower() not in _STOPWORDS] or labels[best].split()
        names[c] = " ".join(_pretty(w) for w in words[:4])
    used = {}
    for c in range(k):
        base, nm, i = names[c], names[c], 2
        while nm in used:
            nm = f"{base} {i}"; i += 1
        used[nm] = True
        names[c] = nm
    return names


def _balance(matrix, assign, k, cap=9, max_k=16):
    """Break up over-large clusters: repeatedly split the biggest cluster in two
    until every cluster is <= cap (or we hit max_k). Fixes the case where one
    dense region swallows most notes into a single vague core."""
    assign = np.array(assign)
    while k < max_k:
        sizes = {c: int((assign == c).sum()) for c in set(assign.tolist())}
        big = max(sizes, key=sizes.get)
        if sizes[big] <= cap:
            break
        idx = np.where(assign == big)[0]
        sub, _ = _kmeans(matrix[idx], 2)
        for j, i in enumerate(idx):
            if sub[j] == 1:
                assign[i] = k
        k += 1
    uniq = sorted(set(assign.tolist()))
    remap = {o: n for n, o in enumerate(uniq)}
    return np.array([remap[int(a)] for a in assign]), len(uniq)


def _merge_small(matrix, assign, k, min_size=3):
    """Fold tiny clusters (1-2 notes) into their nearest real topic, so we don't
    get single-note 'cores'. Renumbers the clusters afterwards."""
    assign = np.array(assign)
    sizes = {c: int((assign == c).sum()) for c in range(k)}
    big = [c for c in range(k) if sizes[c] >= min_size]
    if len(big) < 2:
        return assign, k
    cents = np.vstack([
        (lambda v: v / (np.linalg.norm(v) + 1e-8))(matrix[assign == c].mean(axis=0))
        for c in big
    ])
    for i in range(len(assign)):
        if sizes[int(assign[i])] < min_size:
            assign[i] = big[int((matrix[i] @ cents.T).argmax())]
    uniq = sorted(set(assign.tolist()))
    remap = {old: new for new, old in enumerate(uniq)}
    assign = np.array([remap[int(a)] for a in assign])
    return assign, len(uniq)


# --- Real links between notes (Obsidian-style) --------------------------------
# The memory notes link to each other with [[wikilinks]]. Those are the real
# edges of the graph — exactly what Obsidian draws. We parse them, resolve each
# target to the note it names, and turn them into note-to-note edges.

_WIKILINK = re.compile(r"\[\[([^\]|#]+)")


def _norm_key(s):
    """Collapse a slug/filename to a comparison key: lowercase, letters+digits
    only. So 'side-project-loyalty-app', 'side_project_loyalty_app' and 'Side
    Project Loyalty App' all reduce to the same thing."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


_KEY_PREFIXES = ("project_", "project-", "feedback_", "feedback-",
                 "reference_", "reference-", "user_", "user-")


def _strip_prefix(stem):
    """Return the stem without a leading project_/feedback_/reference_/user_
    prefix, so [[brew-guide]] resolves to reference_brew_guide."""
    low = stem.lower()
    for p in _KEY_PREFIXES:
        if low.startswith(p):
            return stem[len(p):]
    return stem


def _frontmatter(text):
    """The YAML-ish block between the leading --- fences, or '' if there is none."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    return m.group(1) if m else ""


def _frontmatter_name(text):
    """The canonical slug a memory note gives itself in its `name:` frontmatter."""
    m = re.search(r"^name:\s*(.+)$", _frontmatter(text) or text, re.MULTILINE)
    return m.group(1).strip() if m else None


_TYPES = ("project", "feedback", "reference", "user")


def _note_type(text, stem):
    """What kind of note this is: from its frontmatter `type:` if present, else
    from its filename prefix (project_/feedback_/reference_/user_), else 'note'.
    Only looks inside the frontmatter so body text like 'Type = Text' is ignored."""
    m = re.search(r"^\s*type:\s*([a-zA-Z]+)", _frontmatter(text), re.MULTILINE)
    if m:
        t = m.group(1).lower()
        if t in _TYPES:
            return t
    low = stem.lower()
    for t in _TYPES:
        if low.startswith(t):
            return t
    return "note"


def _extract_links(text):
    """Every [[target]] a note points at (drops any |alias or #heading part)."""
    return [m.group(1).strip() for m in _WIKILINK.finditer(text)]


def graph_data(n_topics=8, sim_per_node=3, sim_threshold=0.62):
    """Build an Obsidian-style graph: notes are nodes, and edges are the real
    links between them.

      - LINK edges come from [[wikilinks]] in the note text (the true structure).
      - SIM edges connect each note to its few most semantically-similar notes,
        so notes without explicit links still join the web (like Obsidian's
        Smart Connections). Lighter, and separately toggleable in the UI.
      - Clustering is kept only to COLOUR nodes by topic, not to shape the graph.
      - Each node carries its degree and link-count so the UI can size it.

    Returns {"nodes": [...], "edges": [...]}.
    """
    import ingest
    from collections import defaultdict

    conn = _db()
    rows = conn.execute("SELECT source, embedding FROM chunks").fetchall()
    conn.close()

    per_note = defaultdict(list)
    for source, blob in rows:
        per_note[source].append(np.frombuffer(blob, dtype=np.float32))

    sources = sorted(per_note)
    if not sources:
        return {"nodes": [], "edges": []}

    matrix = np.vstack([
        (lambda v: v / (np.linalg.norm(v) + 1e-8))(np.mean(per_note[s], axis=0))
        for s in sources
    ])

    labels = [_map_label(s.partition("/")[2] or s) for s in sources]
    assign, k = _kmeans(matrix, n_topics)
    assign, k = _balance(matrix, assign, k, cap=9)
    names = _core_names(labels, matrix, assign, k)
    group_of = {s: names[int(assign[i])] for i, s in enumerate(sources)}

    # Read each note off disk once, to pull out wikilinks and build a resolver
    # from "any name it could be referred to by" -> its source id.
    disk = ingest.current_files()  # source -> (path, label, rel, mtime)
    text_of = {}
    type_of = {}
    key_to_src = {}
    label_of = {s: labels[i] for i, s in enumerate(sources)}

    def add_key(key, src):
        key = _norm_key(key)
        if key and key not in key_to_src:
            key_to_src[key] = src

    for s in sources:
        info = disk.get(s)
        text = ""
        if info:
            try:
                text = info[0].read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
        text_of[s] = text
        rel = (info[2] if info else s).replace("\\", "/")
        stem = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        type_of[s] = _note_type(text, stem)
        add_key(stem, s)
        add_key(_strip_prefix(stem), s)
        fn = _frontmatter_name(text)
        if fn:
            add_key(fn, s)
            add_key(_strip_prefix(fn), s)

    edges = []
    seen = set()

    def add_edge(a, b, kind, w=1.0):
        if a == b:
            return
        pair = (a, b) if a < b else (b, a)
        if pair in seen:
            return
        seen.add(pair)
        edges.append({"source": a, "target": b, "kind": kind, "w": round(float(w), 3)})

    # 1. Real links from [[wikilinks]] — kept DIRECTED so the reading panel can
    #    show "links to" (out) and "linked from" (backlinks, in) separately, and
    #    flag links that point at a note that does not exist (broken).
    out_map = defaultdict(list)     # source -> [target sources it links to]
    broken_map = defaultdict(list)  # source -> [raw link text that resolved to nothing]
    for s in sources:
        for target in _extract_links(text_of.get(s, "")):
            tsrc = (key_to_src.get(_norm_key(target))
                    or key_to_src.get(_norm_key(_strip_prefix(target))))
            if tsrc and tsrc != s:
                if tsrc not in out_map[s]:
                    out_map[s].append(tsrc)
                add_edge(s, tsrc, "link")
            elif not tsrc:
                broken_map[s].append(target.strip())

    in_map = defaultdict(list)      # target -> [sources that link to it] (backlinks)
    for s, targets in out_map.items():
        for t in targets:
            in_map[t].append(s)

    # 2. Similarity edges: each note to its top few nearest notes above a floor.
    if sim_per_node > 0 and len(sources) > 1:
        sims = matrix @ matrix.T
        np.fill_diagonal(sims, -1.0)
        for i, s in enumerate(sources):
            order = np.argsort(sims[i])[::-1][:sim_per_node]
            for j in order:
                if sims[i, j] < sim_threshold:
                    break
                add_edge(s, sources[int(j)], "sim", sims[i, j])

    deg = defaultdict(int)
    for e in edges:
        deg[e["source"]] += 1
        deg[e["target"]] += 1

    def link_list(src_ids):
        return [{"id": t, "label": label_of.get(t, t)} for t in src_ids]

    nodes = [{
        "id": s,
        "label": labels[i],
        "group": group_of[s],
        "type": type_of.get(s, "note"),
        "area": area_of(s),
        "proj": s.partition("/")[0],
        "kind": "note",
        "deg": int(deg.get(s, 0)),
        "links": len(out_map.get(s, [])) + len(in_map.get(s, [])),
        "out": link_list(out_map.get(s, [])),
        "in": link_list(in_map.get(s, [])),
        "broken": sorted(set(broken_map.get(s, []))),
    } for i, s in enumerate(sources)]

    projects = sorted({n["proj"] for n in nodes})
    return {
        "nodes": nodes, "edges": edges,
        "groups": list(names.values()), "projects": projects,
        "areas": list_areas(),
    }
