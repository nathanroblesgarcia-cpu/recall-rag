# Recall — a local RAG app over your own notes, with an eval suite

Ask a plain-English question, get an answer grounded in your own notes, with the
source note named. Everything runs on your machine. Your notes never leave it.

Recall is a small but complete example of a **production-minded RAG system**:
retrieval, an agentic tool loop, a local or cloud answer model, and an
**automated eval suite** that scores how good the answers are. It is built to be
read, not just run.

> This public repo ships with a **fictional** sample note set — the working notes
> of a made-up café and coffee-roastery owner ("Sam Carter") — so it runs out of
> the box with no private data. Point it at your own note folders to use it for real.

![Recall screenshot](docs/screenshot.png)

---

## What this project demonstrates

The things AI-engineering roles actually screen for, in one small codebase:

- **RAG (retrieval-augmented generation).** Notes are chunked, embedded, and
  searched by meaning, then the top chunks are handed to a model to answer from.
- **Local-first, private.** Embeddings (fastembed / `bge-small`) and generation
  (Ollama / `llama3.2:3b`) both run on-device. A cloud model (Claude) is a
  config switch, not a requirement.
- **An agentic tool loop.** The answer path can call tools (search, journal
  summary) rather than a single prompt.
- **Evals as a first-class citizen.** A golden question set, a scorer for both
  retrieval and answer accuracy, and an LLM-as-judge pass. This is the part that
  separates a demo from something you would trust.
- **Readable engineering.** Every module is commented in plain language, config
  is centralised, and the DB is a single SQLite file.

## How it works, in one picture

```
your question
   -> embed the question into 384 numbers (bge-small, local)
   -> compare against every stored note chunk (cosine similarity)
   -> take the closest chunks
   -> hand them to the answer model with the question
   -> model answers using only those chunks, and cites the source note
```

Three moving parts:

1. **Ingest** (`ingest.py`) — read notes, cut into chunks, embed, save to SQLite.
2. **Retrieve** (`rag.py`) — embed the question, find the closest chunks.
3. **Generate** (`rag.py`) — the model writes the answer from those chunks.

## Results (on the bundled 16-question demo set)

Run `python evals/run_eval.py` to reproduce these on your machine.

| Metric | Score |
|---|---|
| Retrieval hit rate (right note pulled in) | **16/16 (100%)** |
| Answer pass rate (all required facts present) | **16/16 (100%)** |

Measured with the local `llama3.2:3b` answer model on the 16-question demo set.
The demo corpus is deliberately clean, so a perfect score here means the pipeline
is wired correctly end to end; on a larger, messier real corpus the interesting
work is watching this number and pushing it back up after each change.

The eval writes a timestamped report to `evals/reports/` each run, so you can see
whether a change made the score go up or down. See [`evals/README.md`](evals/README.md).

## Run it yourself

Requires Python 3.12 and (for local answers) [Ollama](https://ollama.com).

```bash
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python ingest.py        # build the index from sample_notes/ (first run downloads bge-small, ~130 MB)
python app.py           # open http://127.0.0.1:5170
```

Prefer no local model? Set `GEN_BACKEND = "none"` in `config.py` to see retrieval
only, or `"anthropic"` with an `ANTHROPIC_API_KEY` to answer with Claude.

## Run the evals

```bash
python evals/run_eval.py                  # retrieval + answer
python evals/run_eval.py --retrieval-only # fast, no answer model needed
```

## Repo layout

| Path | What it is |
|---|---|
| `ingest.py`, `indexer.py` | build and update the search index |
| `rag.py` | retrieval + answer generation (the core) |
| `agent.py`, `tools.py` | the agentic tool loop |
| `app.py`, `templates/`, `static/` | the Flask web page |
| `evals/` | golden set, scorer, LLM-as-judge, reports |
| `sample_notes/` | the fictional demo corpus |
| `config.py` | every setting in one place |

## Use it on your own notes

Edit `CORPUS_FOLDERS` in `config.py` to point at your own folders of `.md` notes,
set `USER_NAME`, and re-run `python ingest.py`. Your notes stay local and are
gitignored by default.

## Design notes on privacy

Retrieval always runs locally. With the default Ollama backend, generation is
local too, so a question and its answer never leave the machine. The built index
(`data/`) and any notes you type in (`my_notes/`) are gitignored, so a personal
deployment never commits private content.

## License

MIT. See [LICENSE](LICENSE).
