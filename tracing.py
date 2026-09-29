# tracing.py
# A record of every step behind one answer, so a wrong or slow answer can be
# looked at instead of guessed at. Same idea as a browser's network tab.
#
# One question = one trace = one line in data/traces.jsonl:
#   what was asked, which notes were found (and their scores), each model round
#   (time + tokens in/out), each tool slip and what came back, any guard that
#   fired, and the final answer.
#
# The file is plain append-only text (not SQLite), kept next to the index in
# data/. data/ is gitignored, so traces never reach GitHub.
# View them at /traces.

import json
import time
import uuid
from datetime import datetime

import config

TRACE_FILE = config.BASE_DIR / "data" / "traces.jsonl"
KEEP = 500          # keep the newest N traces; older ones are trimmed away
PREVIEW = 300       # how much of long text (answers, tool results) to keep


def _short(value, limit=PREVIEW):
    """Trim long text so one trace stays small and readable."""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return value if len(value) <= limit else value[:limit] + " ..."


class Trace:
    """Collects the steps of one question, then writes them as one line.

    Use:
        tr = Trace(question, mode="notes")
        with tr.step("search") as s:
            ...do the work...
            s["sources"] = [...]        # add whatever is worth seeing
        tr.finish(answer)
    """

    def __init__(self, question, mode, model=None):
        self.data = {
            "id": uuid.uuid4().hex[:10],
            "at": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "model": model,
            "question": question,
            "steps": [],
        }
        self._t0 = time.perf_counter()

    def step(self, kind, **fields):
        """Time one step. Returns a context manager that yields a dict you can
        add fields to; the step is saved (with its ms) when the block ends,
        even if it raised."""
        trace = self

        class _Step:
            def __enter__(self_inner):
                self_inner.rec = {"kind": kind, **fields}
                self_inner.t = time.perf_counter()
                return self_inner.rec

            def __exit__(self_inner, exc_type, exc, tb):
                self_inner.rec["ms"] = round((time.perf_counter() - self_inner.t) * 1000)
                if exc is not None:
                    self_inner.rec["error"] = str(exc)
                trace.data["steps"].append(self_inner.rec)
                return False  # never swallow the error

        return _Step()

    def note(self, kind, **fields):
        """Record an instant event with no timing (e.g. a guard firing)."""
        self.data["steps"].append({"kind": kind, **fields})

    def finish(self, answer=None, error=None):
        """Add the totals and write the trace. Never raises: a logging hiccup
        must never break an answer."""
        d = self.data
        d["total_ms"] = round((time.perf_counter() - self._t0) * 1000)
        d["tokens_in"] = sum(s.get("tokens_in") or 0 for s in d["steps"])
        d["tokens_out"] = sum(s.get("tokens_out") or 0 for s in d["steps"])
        d["answer"] = _short(answer or "", 2000)
        if error:
            d["error"] = str(error)
        try:
            TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(TRACE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
            _trim()
        except Exception as e:
            print(f"[tracing] could not write trace: {e}")


def _trim():
    """Keep only the newest KEEP traces, so the file never grows forever."""
    lines = TRACE_FILE.read_text(encoding="utf-8").splitlines()
    if len(lines) > KEEP * 1.2:  # trim in batches, not on every write
        TRACE_FILE.write_text("\n".join(lines[-KEEP:]) + "\n", encoding="utf-8")


def recent(limit=100):
    """Newest traces first, for the /traces page."""
    if not TRACE_FILE.exists():
        return []
    out = []
    for line in TRACE_FILE.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # a half-written line: skip it
    return out[::-1]


def _ms(ns):
    return round(ns / 1e6) if ns else None


def usage_from(ollama_json):
    """Pull token counts and the time split out of an Ollama reply (the final
    one when streaming). Ollama reports three separate clocks:
      load_ms    loading the model into memory (0 when it is already loaded)
      read_ms    reading everything on the desk (the tokens in)
      write_ms   writing the answer (the tokens out)"""
    return {
        "tokens_in": ollama_json.get("prompt_eval_count"),
        "tokens_out": ollama_json.get("eval_count"),
        "load_ms": _ms(ollama_json.get("load_duration")),
        "read_ms": _ms(ollama_json.get("prompt_eval_duration")),
        "write_ms": _ms(ollama_json.get("eval_duration")),
    }
