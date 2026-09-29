# app.py
# The web page. A simple single-user Flask app.
#   GET  /      shows the question box
#   POST /ask   runs the RAG pipeline and shows the answer + sources
#   POST /ask_agent  notes + live numbers (the agent), streamed step by step
#   GET  /traces     the record of every step behind recent answers

import json
import os
import threading
import time

from flask import Flask, render_template, request, jsonify, Response, redirect

import config
import indexer
import ingest
import journal
import rag
import tracing

app = Flask(__name__)

# Remembers the recent turns so follow-up questions work. Single-user local app,
# so a simple in-memory list is enough. "Start fresh" clears it.
HISTORY = []


@app.context_processor
def inject_areas():
    """Make the life-area list available to every template (for the 'Ask about…'
    picker) without threading it through each render_template call."""
    return {"areas": rag.list_areas()}


def _refresh_message(delta):
    """Turn an update_index result into a headline + named lists for the page,
    so a refresh says exactly which notes were added, updated, or removed."""
    n = len(delta["added"]) + len(delta["changed"])
    if not n and not delta["removed"]:
        headline = "Already up to date — nothing changed."
    else:
        bits = []
        if delta["added"]:
            bits.append(f"{len(delta['added'])} added")
        if delta["changed"]:
            bits.append(f"{len(delta['changed'])} updated")
        if delta["removed"]:
            bits.append(f"{len(delta['removed'])} removed")
        headline = "Refreshed: " + ", ".join(bits) + "."

    # Tag each added/updated note with the area it landed in, so the page can show
    # an editable area dropdown right beside the note that just changed. Removed
    # notes are gone, so they need no area.
    def with_area(items):
        return [{**it, "area": rag.area_of(it["source"])} for it in items]
    return {
        "headline": headline,
        "added": with_area(delta["added"]),
        "changed": with_area(delta["changed"]),
        "removed": delta["removed"],
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, has_history=bool(HISTORY), saved=None, refreshed=None)


@app.route("/models")
def models():
    """List the local models the user can pick from, plus the default."""
    names, current = rag.list_models()
    return jsonify({"models": names, "current": current})


@app.route("/ask", methods=["POST"])
def ask():
    question = (request.form.get("question") or "").strip()
    if not question:
        return render_template("index.html", result=None, has_history=bool(HISTORY), saved=None, refreshed=None)
    model = (request.form.get("model") or "").strip() or None
    result = rag.answer_question(question, history=HISTORY, model=model)
    HISTORY.append({"question": question, "answer": result["answer"] or ""})
    del HISTORY[:-3]  # keep only the last 3 turns
    return render_template("index.html", result=result, has_history=bool(HISTORY), saved=None, refreshed=None)


@app.route("/add", methods=["POST"])
def add():
    title = (request.form.get("title") or "").strip()
    note = (request.form.get("note") or "").strip()
    if not note:
        return render_template("index.html", result=None, has_history=bool(HISTORY), saved=None, refreshed=None)
    ingest.add_note(title, note, rag._embedder)
    return render_template(
        "index.html", result=None, has_history=bool(HISTORY), saved=(title or "your note"), refreshed=None
    )


@app.route("/ask_stream", methods=["POST"])
def ask_stream():
    """Stream the answer to the page as the model writes it. The first line is a
    JSON header with the retrieved notes; everything after it is answer text."""
    question = (request.form.get("question") or "").strip()
    if not question:
        return Response("", mimetype="text/plain")

    model = (request.form.get("model") or "").strip() or None
    area = (request.form.get("area") or "").strip() or None
    tr = tracing.Trace(question, mode="notes", model=model or config.OLLAMA_MODEL)
    with tr.step("search", area=area) as rec:
        retrieved = rag.retrieve(question, history=HISTORY, area=area)
        rec["found"] = [{"source": r["source"], "score": round(r["score"], 3)} for r in retrieved]
    header = json.dumps({
        "sources": [
            {"source": r["source"], "score": r["score"], "text": r["text"]}
            for r in retrieved
        ]
    })

    def gen():
        yield header + "\n"
        collected = []
        usage = {}
        error = None
        try:
            with tr.step("model", round=1) as rec:
                start = time.perf_counter()
                for delta in rag.generate_stream(question, retrieved, history=HISTORY, model=model, usage=usage):
                    if not collected:
                        rec["first_word_ms"] = round((time.perf_counter() - start) * 1000)
                    collected.append(delta)
                    yield delta
                rec.update(usage)
        except Exception as exc:
            error = exc
            raise
        finally:
            tr.finish("".join(collected), error=error)
        HISTORY.append({"question": question, "answer": "".join(collected)})
        del HISTORY[:-3]

    return Response(gen(), mimetype="text/plain; charset=utf-8")


@app.route("/ask_agent", methods=["POST"])
def ask_agent():
    """Notes + live numbers mode (v1.19): the agent (agent.py) decides whether to
    search notes, call the café number tools (cafe_tools.py), or both. It is slow on CPU
    (a 7B model, several rounds), so each step is streamed to the page as one
    JSON line the moment it happens, then a final {"type": "answer"} line.
    Stand-alone questions only: follow-up memory belongs to Notes-only mode."""
    import queue
    import agent

    question = (request.form.get("question") or "").strip()
    if not question:
        return Response("", mimetype="application/x-ndjson")

    events = queue.Queue()

    def work():
        tr = tracing.Trace(question, mode="agent", model=agent.AGENT_MODEL)
        answer, error = None, None
        try:
            out = agent.ask(question, verbose=False, on_step=events.put, trace=tr)
            answer = out["answer"]
            events.put({"type": "answer", "answer": answer})
        except Exception as exc:
            error = exc
            events.put({"type": "error", "error": str(exc)})
        finally:
            tr.finish(answer, error=error)
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def gen():
        while True:
            ev = events.get()
            if ev is None:
                break
            yield json.dumps(ev, ensure_ascii=False, default=str) + "\n"

    return Response(gen(), mimetype="application/x-ndjson; charset=utf-8")


@app.route("/traces")
def traces_view():
    """The record behind recent answers: every step, its time and its tokens."""
    return render_template("traces.html", traces=tracing.recent())


@app.route("/refresh", methods=["POST"])
def refresh():
    # Kick off a background re-index and return immediately, so the page never
    # freezes while embedding runs. It pulls the read-only mirrors (CPI client
    # knowledge — pull-only, never writes to the source repos) then re-embeds only
    # what changed on disk. The page polls /refresh_status for progress. If a build
    # is already running (e.g. the one started at launch), we just report that and
    # the page attaches to it.
    started = indexer.start(rag._embedder)
    return jsonify({"started": started, "already_running": not started})


@app.route("/refresh_status")
def refresh_status():
    """Progress of the background index, polled by the page. When a build has
    finished, also hand back the rendered 'what changed' block and the fresh
    category breakdown so the page can drop them in without a reload."""
    st = indexer.status()
    payload = {
        "running": st["running"],
        "phase": st["phase"],
        "done": st["done"],
        "total": st["total"],
        "eta": st["eta"],
        "message": st["message"],
        "error": st["error"],
    }
    if st["phase"] == "done" and st["delta"] is not None:
        refreshed = _refresh_message(st["delta"])
        refreshed["mirrors"] = st["mirrors"]
        payload["refreshed_html"] = render_template("_refreshed.html", refreshed=refreshed)
        payload["categories_html"] = render_template(
            "_categories.html", categories=rag.categorised_notes()
        )
    return jsonify(payload)


@app.route("/categories")
def categories():
    """The category breakdown as a standalone fragment, so the page can re-render
    it live after a note's area is changed without a full reload."""
    return render_template("_categories.html", categories=rag.categorised_notes())


@app.route("/set_area", methods=["POST"])
def set_area():
    """Manually file one note into a life area. The choice is saved and overrides
    the automatic sorting."""
    source = (request.form.get("source") or "").strip()
    area = (request.form.get("area") or "").strip()
    if not source or not area:
        return jsonify({"ok": False, "error": "missing source or area"}), 400
    saved = rag.set_area_override(source, area)
    if saved is None:
        return jsonify({"ok": False, "error": "unknown area"}), 400
    return jsonify({"ok": True, "source": source, "area": saved})


@app.route("/fresh", methods=["POST"])
def fresh():
    HISTORY.clear()
    return render_template("index.html", result=None, has_history=False, saved=None, refreshed=None)


@app.route("/map")
def map_view():
    return render_template("map.html", graph=rag.graph_data())


@app.route("/note")
def note():
    source = request.args.get("source", "")
    return jsonify({"source": source, "text": rag.note_text(source)})


@app.route("/journal")
def journal_view():
    return render_template("journal.html", days=journal.journal_days())


@app.route("/journal/note", methods=["POST"])
def journal_note():
    journal.set_note(request.form.get("date", ""), request.form.get("note", ""))
    return redirect("/journal")


@app.route("/journal/summary", methods=["POST"])
def journal_summary():
    date = request.form.get("date", "")
    return jsonify({"date": date, "summary": journal.make_summary(date)})


if __name__ == "__main__":
    # Pull the read-only mirrors (pull-only, never writes to the source repos),
    # then re-embed only the notes that changed since last time (and drop deleted
    # ones). Fast, because unchanged notes are untouched.
    #
    # Guarded: if another index build is already running (it holds the database),
    # startup indexing would hit 'database is locked'. Rather than fail to launch,
    # skip it and serve what's already indexed — the running build keeps going and
    # a later Refresh picks up the rest.
    #
    # RECALL_SKIP_STARTUP_INDEX=1 turns startup indexing off entirely. Set it when
    # a separate bulk build is the sole writer, so the app serves as a pure reader
    # and never competes for the write-lock.
    if os.environ.get("RECALL_SKIP_STARTUP_INDEX"):
        print("Startup indexing disabled (RECALL_SKIP_STARTUP_INDEX) — serving read-only.")
    else:
        # Index in the BACKGROUND so the app binds its port and serves the page
        # immediately, even when the corpus needs a big re-embed. Progress shows in
        # the page's Refresh bar; the existing index answers questions meanwhile.
        indexer.start(rag._embedder)
        print("Startup indexing running in the background — the app is ready now.")
    # Preload the local model in the background so the first question is fast.
    # RECALL_SKIP_WARM=1 skips this — used by the boot autostart so a machine that
    # just turned on isn't holding ~3 GB for Ollama before the user has asked
    # anything. The model then loads on the first question instead (a few seconds
    # slower, once). When the app is opened by hand we DO warm, so answers are snappy.
    if not os.environ.get("RECALL_SKIP_WARM"):
        threading.Thread(target=rag.warm, daemon=True).start()

    # Honour a PORT override from the environment (used when a preview harness
    # assigns a free port); fall back to the configured default otherwise.
    port = int(os.environ.get("PORT", config.PORT))
    print(f"Recall running at http://127.0.0.1:{port}")
    # use_reloader=False so the startup rebuild runs exactly once.
    # threaded=True so streaming an answer doesn't block the page's other requests.
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False, threaded=True)
