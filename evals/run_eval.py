# run_eval.py
# A graded quiz for Recall.
#
# It asks Recall a fixed set of questions whose correct answers you already know
# (the "golden set" in golden.json), then scores two things independently:
#
#   1. RETRIEVAL   Did the right note get pulled into the sources?   (objective)
#   2. ANSWER      Did the written answer contain the known facts?   (fact-presence)
#
# Retrieval is the core RAG metric and needs no answer model, so --retrieval-only
# runs in seconds without Ollama. A full run also grades the written answers.
#
# USAGE (from the Recall folder, with its venv active):
#     python evals\run_eval.py                    # full: retrieval + answer
#     python evals\run_eval.py --retrieval-only   # fast: retrieval only, no Ollama
#     python evals\run_eval.py --model qwen2.5:7b # grade the "Smarter" model
#     python evals\run_eval.py --limit 5          # first 5 questions only
#
# Reports are written to evals\reports\ as a timestamped .md and .json so you can
# keep a history and see whether a change to Recall made the score go up or down.
#
# NOTE ON TOOLING: the industry-standard eval runners are Promptfoo (Node) and
# LangSmith / Braintrust (hosted). This is a deliberately small pure-Python runner
# so it reuses Recall's own venv - no Node install, no cloud account, nothing
# leaves the machine. The concepts are identical: a golden dataset, a scorer, and
# a pass rate you can quote. NEXT STEP for the AI-engineering plan: replace the
# substring fact-check with an LLM-as-judge, so a correct answer phrased
# differently ("sixty-five degrees" vs "65") still scores as correct.

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# This file lives in Recall/evals/. Add the Recall folder (its parent) to the
# import path so we can import the app's own rag/config modules and call the
# exact same pipeline the web page uses.
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config  # noqa: E402
import rag      # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.json"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def _norm(s):
    """Lowercase for case-insensitive matching. None becomes empty string."""
    return (s or "").lower()


def fact_present(answer, fact):
    """A fact is a plain string (that substring must appear) or a list of options
    (ANY one of them appearing counts as a match - used for number-format or
    wording variants). Returns True/False."""
    a = _norm(answer)
    if isinstance(fact, list):
        return any(_norm(opt) in a for opt in fact)
    return _norm(fact) in a


def score_answer(answer, expected_facts):
    """Fraction of required facts present in the answer, plus the list of any that
    were missing (for the report). No expected facts => coverage 1.0."""
    if not expected_facts:
        return 1.0, []
    missing = [f for f in expected_facts if not fact_present(answer, f)]
    covered = len(expected_facts) - len(missing)
    return covered / len(expected_facts), missing


def score_retrieval(sources, expected_sources):
    """Did any retrieved source label contain any expected substring? Returns
    (hit, rank, label, scorevalue): rank is the 1-based position of the first
    matching source (lower is better), or None on a miss."""
    for i, s in enumerate(sources, start=1):
        label = _norm(s.get("source"))
        for exp in expected_sources:
            if _norm(exp) in label:
                return True, i, s.get("source"), s.get("score")
    return False, None, None, None


def run(items, retrieval_only, model):
    """Ask each question, score it, and return a list of per-item result dicts."""
    results = []
    for n, item in enumerate(items, start=1):
        q = item["question"]
        print(f"[{n}/{len(items)}] {item['id']}: {q}")
        t0 = time.time()
        if retrieval_only:
            sources = rag.retrieve(q)
            answer = None
        else:
            out = rag.answer_question(q, model=model)
            sources = out["sources"]
            answer = out["answer"]
        secs = time.time() - t0

        hit, rank, hit_label, hit_score = score_retrieval(sources, item["expected_sources"])
        if retrieval_only:
            coverage, missing = None, None
        else:
            coverage, missing = score_answer(answer, item["expected_facts"])

        results.append({
            "id": item["id"],
            "question": q,
            "tests": item.get("tests", ""),
            "expected_sources": item["expected_sources"],
            "expected_facts": item["expected_facts"],
            "retrieval_hit": hit,
            "retrieval_rank": rank,
            "retrieval_label": hit_label,
            "retrieval_score": hit_score,
            "top_sources": [s.get("source") for s in sources[:5]],
            "answer": answer,
            "answer_coverage": coverage,
            "answer_missing": missing,
            "seconds": round(secs, 2),
        })
    return results


def summarise(results, retrieval_only):
    """Roll the per-item results up into the headline numbers."""
    total = len(results)
    hits = sum(1 for r in results if r["retrieval_hit"])
    summary = {
        "total_questions": total,
        "retrieval_hits": hits,
        "retrieval_hit_rate": round(hits / total, 3) if total else 0.0,
        "total_seconds": round(sum(r["seconds"] for r in results), 1),
    }
    if not retrieval_only:
        # Answer PASS = every required fact present (coverage 1.0). Coverage is the
        # softer average across all facts, so a near-miss still shows partial credit.
        passed = sum(1 for r in results if r["answer_coverage"] == 1.0)
        avg_cov = sum(r["answer_coverage"] for r in results) / total if total else 0.0
        summary["answer_passes"] = passed
        summary["answer_pass_rate"] = round(passed / total, 3) if total else 0.0
        summary["answer_avg_coverage"] = round(avg_cov, 3)
    return summary


def render_markdown(summary, results, retrieval_only, model, when):
    """Human-readable report: the headline numbers you can quote, then a per-question
    table so you can see exactly what missed."""
    lines = []
    lines.append(f"# Recall eval report - {when}")
    lines.append("")
    mode = "retrieval only" if retrieval_only else f"full (answer model: {model or config.OLLAMA_MODEL})"
    lines.append(f"Mode: {mode}  |  Questions: {summary['total_questions']}  |  Time: {summary['total_seconds']}s")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Retrieval hit rate: **{summary['retrieval_hits']}/{summary['total_questions']}** "
                 f"({summary['retrieval_hit_rate']*100:.0f}%) - the right note was pulled in")
    if not retrieval_only:
        lines.append(f"- Answer pass rate: **{summary['answer_passes']}/{summary['total_questions']}** "
                     f"({summary['answer_pass_rate']*100:.0f}%) - every required fact present")
        lines.append(f"- Answer fact coverage: **{summary['answer_avg_coverage']*100:.0f}%** - "
                     f"average share of facts present (partial credit)")
    lines.append("")
    lines.append("## Per question")
    lines.append("")
    if retrieval_only:
        lines.append("| Question | Retrieval | Rank | Secs |")
        lines.append("|---|---|---|---|")
        for r in results:
            ret = "PASS" if r["retrieval_hit"] else "MISS"
            rank = r["retrieval_rank"] if r["retrieval_rank"] else "-"
            lines.append(f"| {r['id']} | {ret} | {rank} | {r['seconds']} |")
    else:
        lines.append("| Question | Retrieval | Rank | Answer facts | Secs |")
        lines.append("|---|---|---|---|---|")
        for r in results:
            ret = "PASS" if r["retrieval_hit"] else "MISS"
            rank = r["retrieval_rank"] if r["retrieval_rank"] else "-"
            n_facts = len(r["expected_facts"])
            n_have = n_facts - len(r["answer_missing"]) if r["answer_missing"] is not None else n_facts
            ans = f"{n_have}/{n_facts}"
            lines.append(f"| {r['id']} | {ret} | {rank} | {ans} | {r['seconds']} |")
    lines.append("")

    # Detail only for the ones that missed, so the report leads with what to fix.
    problems = [r for r in results if not r["retrieval_hit"]
                or (not retrieval_only and r["answer_coverage"] != 1.0)]
    if problems:
        lines.append("## What missed")
        lines.append("")
        for r in problems:
            lines.append(f"### {r['id']} - {r['question']}")
            if not r["retrieval_hit"]:
                lines.append(f"- RETRIEVAL MISS. Wanted a source matching {r['expected_sources']}. "
                             f"Top sources were: {r['top_sources']}")
            if not retrieval_only and r["answer_missing"]:
                lines.append(f"- ANSWER missing facts: {r['answer_missing']}")
                snippet = (r["answer"] or "").strip().replace("\n", " ")
                if len(snippet) > 300:
                    snippet = snippet[:300] + "..."
                lines.append(f"- Answer said: {snippet}")
            lines.append("")
    else:
        lines.append("All questions passed retrieval"
                     + ("" if retrieval_only else " and answer") + ". Clean run.")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Run the Recall eval quiz.")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="Score retrieval only - fast, no answer model needed.")
    ap.add_argument("--model", default=None,
                    help="Override the local answer model (e.g. qwen2.5:7b).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N questions.")
    args = ap.parse_args()

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    items = golden["items"]
    if args.limit:
        items = items[:args.limit]

    print(f"Recall eval - {len(items)} questions, "
          f"{'retrieval only' if args.retrieval_only else 'full'} mode\n")

    results = run(items, args.retrieval_only, args.model)
    summary = summarise(results, args.retrieval_only)

    print("\n" + "=" * 50)
    print(f"Retrieval hit rate: {summary['retrieval_hits']}/{summary['total_questions']} "
          f"({summary['retrieval_hit_rate']*100:.0f}%)")
    if not args.retrieval_only:
        print(f"Answer pass rate:   {summary['answer_passes']}/{summary['total_questions']} "
              f"({summary['answer_pass_rate']*100:.0f}%)")
        print(f"Answer coverage:    {summary['answer_avg_coverage']*100:.0f}%")
    print(f"Total time:         {summary['total_seconds']}s")
    print("=" * 50)

    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = "retrieval" if args.retrieval_only else "full"

    md = render_markdown(summary, results, args.retrieval_only, args.model, when)
    md_path = REPORTS_DIR / f"{stamp}_{tag}.md"
    md_path.write_text(md, encoding="utf-8")

    json_path = REPORTS_DIR / f"{stamp}_{tag}.json"
    json_path.write_text(json.dumps({"summary": summary, "results": results},
                                     indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nReport: {md_path}")
    print(f"Data:   {json_path}")


if __name__ == "__main__":
    main()
