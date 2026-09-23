# config.py
# All the knobs for the app live here, so you never hunt through code to change a setting.
#
# DEMO BUILD. This public version reads ONLY the bundled sample_notes/ folder, so
# anyone can clone the repo and run it out of the box with fictional data. The
# private build reads the author's real Claude Code memory and note folders; none
# of that is in this repo.

from pathlib import Path

# Where this project lives on disk.
BASE_DIR = Path(__file__).resolve().parent

# The SQLite file that holds your chunks and their meaning-numbers.
# Think of it like the model behind a dashboard: build it once, query it many times.
DB_PATH = BASE_DIR / "data" / "index.db"

# WHICH NOTES TO READ.
# The bundled sample notes: a small, fully fictional corpus so the app runs out of
# the box. Swap CORPUS_FOLDERS below for your own note folders to use it for real.
SAMPLE_NOTES_DIR = BASE_DIR / "sample_notes"

# Notes you type straight into the page get saved here (gitignored).
MY_NOTES_DIR = BASE_DIR / "my_notes"

# Claude Code memory scan is disabled in the demo (this path does not exist, so it
# is skipped). In the private build this points at C:\Users\<you>\.claude\projects
# and every project's memory folder joins the corpus automatically.
CLAUDE_PROJECTS_ROOT = BASE_DIR / "_disabled_projects_scan"

CORPUS_FOLDERS = [
    SAMPLE_NOTES_DIR,   # the bundled fictional demo corpus
    MY_NOTES_DIR,       # notes you type into the page
]

# Folders never to descend into when reading notes, so we never accidentally
# index library files, dependencies, or version-control internals.
SKIP_DIRS = {"venv", ".venv", "site-packages", "node_modules", ".git", "__pycache__"}

# READ-ONLY GIT MIRRORS to fetch before a refresh. None in the demo.
GIT_MIRRORS = []

# WHO THE USER IS.
# So the brain understands that "I", "me", "my" in a question mean this person,
# and treats notes about them as being about the user. Used in the answer prompt.
# This is the demo's fictional persona.
USER_NAME = "Sam Carter"
USER_FIRST_NAME = "Sam"

# LIFE AREAS.
# Every note is sorted into exactly ONE area from its source label, so the page
# can offer an "Ask about…" filter that searches just one slice of your brain.
# Rules are checked top to bottom; the FIRST area whose patterns match the source
# wins. Whatever matches nothing falls to the last area (the catch-all). Patterns
# are lowercase substrings matched against the full source label
# (e.g. "sample_notes/supplier_highland.md").
AREAS = [
    ("Me",            ["me_profile"]),
    ("Money",         ["money_"]),
    ("Personal",      ["personal_"]),
    ("Side projects", ["side_project_"]),
    ("Team",          ["team_"]),
    ("Suppliers",     ["supplier_"]),
    ("Recipes",       ["reference_"]),
    ("Shop",          []),   # catch-all (ops_ and anything else)
]

# THE EMBEDDING MODEL (turns text into meaning-numbers).
# This one runs fully on your machine. Nothing is sent to the cloud.
# 384 numbers per chunk, small and fast, good enough for notes.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 384

# NOTES TO SKIP during indexing.
# MEMORY.md is just an index/summary of all the other notes, so it kept winning
# searches and crowding out the real note. Skipping it loses nothing.
SKIP_FILES = ["MEMORY.md"]

# CHUNKING.
# A chunk is one bite-sized piece of a note. Too big and retrieval gets vague,
# too small and it loses context. ~800 characters is a sensible middle.
CHUNK_CHAR_TARGET = 800

# RETRIEVAL.
# How many of the closest chunks to find and show.
TOP_K = 8

# HYBRID SEARCH (v1.15). Run meaning-search AND word-search (SQLite FTS5), then
# merge the two ranked lists. A chunk ranked high in both rises to the top.
# Set HYBRID_SEARCH = False to go back to meaning-only (handy for A/B evals).
HYBRID_SEARCH = True
# How deep each list goes before merging. 20 beat 50 in testing: deep
# word-search results are mostly common-word noise.
HYBRID_POOL = 20
# The merge is Reciprocal Rank Fusion: each list gives a chunk 1 / (RRF_K + rank).
# 60 is the standard value from the original paper; smaller = top ranks count more.
RRF_K = 60
# How much the word-search list counts in the merge, vs 1.0 for meaning-search.
# 0.7 = words break ties and rescue exact names and codes, meaning still leads.
# Picked from a small settings sweep on the eval (see evals/README.md).
KEYWORD_WEIGHT = 0.7

# RERANKING (v1.17). After search, a small model re-reads the top RERANK_POOL
# chunks alongside the question and re-sorts them; the best TOP_K go on.
# OFF by decision: in testing it nudged rank quality up a little but did not
# change which notes reached the answer model's top 3, and made each search
# much slower. Worth it only when the right note is NOT
# already reaching the answer model. Turn on and re-run the eval to check.
RERANK = False
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANK_POOL = 20

# How many of those chunks to actually feed the answer model, and how much of
# each. More/longer context = the model can actually see the answer even when
# it is spread across several notes.
GEN_CHUNKS = 6
GEN_CHUNK_CHARS = 800

# Keep the local model loaded in memory between questions, and give it enough
# context room for the prompt.
OLLAMA_KEEP_ALIVE = "30m"
OLLAMA_NUM_CTX = 8192

# GENERATION SAMPLING.
# Low temperature keeps the answer glued to the notes instead of inventing
# connective tissue. Grounded fact-retrieval wants this cold, not creative.
OLLAMA_TEMPERATURE = 0.2

# JOURNAL "Summarise this day".
SUMMARY_MAX_NOTES = 15      # at most this many notes per day summary
SUMMARY_CHAR_BUDGET = 6000  # total note text fed, split across the notes used

# WHICH ENGINE WRITES THE ANSWER.
#   "ollama"    = a model running locally on your machine. Free, private, offline.
#   "anthropic" = Claude in the cloud. Costs a fraction of a cent per question,
#                 needs ANTHROPIC_API_KEY in your .env.
#   "none"      = skip the written answer, just show the retrieved notes.
GEN_BACKEND = "ollama"

# OLLAMA (local, free) settings. Ollama runs a small server on your machine;
# this app just sends it the prompt over localhost. Nothing leaves the computer.
OLLAMA_URL = "http://localhost:11434/api/chat"
# Default answer model. A 3B model answers in a few seconds on CPU where a 7B
# takes over a minute, so the 3B is the default "Faster" pick and the 7B is the
# opt-in "Smarter" choice in the page's model picker.
OLLAMA_MODEL = "llama3.2:3b"

# THE JUDGE MODEL (used by the eval's LLM-as-judge, evals/judge.py).
# Grading is a smaller job than writing an answer, so the sharper 7B is affordable
# here and its judgement is more trustworthy than the 3B's. Stays local.
JUDGE_MODEL = "qwen2.5:7b"

# ANTHROPIC (cloud, paid) settings. Used only when GEN_BACKEND = "anthropic".
GEN_MODEL = "claude-haiku-4-5-20251001"

# The web page runs here.
PORT = 5170
