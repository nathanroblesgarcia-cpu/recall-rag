# rejudge.py
# Re-grade a saved eval run's answers with the LLM-as-judge, and show where the
# judge DISAGREES with the original substring scorer.
#
# Why re-grade saved answers instead of running fresh? Because it isolates the one
# thing under test - the GRADER. Same answers, two scorers, side by side. The
# disagreements are the whole point: each one is either the judge rescuing a
# correct-but-reworded answer (the substring scorer's known weakness) or the judge
# making a mistake of its own (which tells you not to trust it yet).
#
# USAGE (from the Recall folder, with its venv):
#     python evals\rejudge.py                       # re-grade the newest full run
#     python evals\rejudge.py --from evals\reports\2026-08-27_170514_full.json
#     python evals\rejudge.py --judge-model llama3.2:3b   # try a weaker judge
#
# Writes a *_judged.md and *_judged.json next to the source report.

import argparse
import glob
import json
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# Put the Recall root on the path (for config) and this evals folder (for judge).
sys.path.insert(0, str(BASE_DIR.parent))
sys.path.insert(0, str(BASE_DIR))

import judge as J  # noqa: E402

REPORTS_DIR = BASE_DIR / "reports"


def substring_verdict(coverage):
    """Map the original run's numeric coverage onto the judge's vocabulary so the
    two graders can be compared on the same three-way scale."""
    if coverage == 1.0:
        return "correct"
    if coverage and coverage > 0:
        return "partial"
    return "incorrect"


def newest_full_report():
    files = sorted(glob.glob(str(REPORTS_DIR / "*_full.json")))
    if not files:
        raise SystemExit("No *_full.json report found in evals/reports/. Run run_eval.py first.")
    return files[-1]


def main():
    ap = argparse.ArgumentParser(description="Re-grade a saved run with the LLM judge.")
    ap.add_argument("--from", dest="src", default=None, help="Path to a *_full.json report. Defaults to the newest.")
    ap.add_argument("--judge-model", default=None, help="Override the judge model (default config.JUDGE_MODEL).")
    args = ap.parse_args()

    src = args.src or newest_full_report()
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    results = data["results"]
    print(f"Re-grading {len(results)} answers from {Path(src).name}")
    print(f"Judge model: {args.judge_model or J.config.JUDGE_MODEL}\n")

    rows = []
    agree = 0
    t0 = time.time()
    for n, r in enumerate(results, start=1):
        req = J.requirement_of(r)
        print(f"[{n}/{len(results)}] {r['id']}")
        jv = J.judge(r["question"], req, r.get("answer"), model=args.judge_model)
        sv = substring_verdict(r.get("answer_coverage"))
        same = jv["verdict"] == sv
        if same:
            agree += 1
        rows.append({
            "id": r["id"],
            "question": r["question"],
            "requirement": req,
            "answer": r.get("answer"),
            "substring": sv,
            "judge": jv["verdict"],
            "judge_reason": jv["reason"],
            "agree": same,
        })
    secs = time.time() - t0

    judge_correct = sum(1 for x in rows if x["judge"] == "correct")
    sub_correct = sum(1 for x in rows if x["substring"] == "correct")
    disagreements = [x for x in rows if not x["agree"]]
    errors = [x for x in rows if x["judge"] == "error"]

    print("\n" + "=" * 54)
    print(f"Substring pass (correct): {sub_correct}/{len(rows)}")
    print(f"Judge pass (correct):     {judge_correct}/{len(rows)}")
    print(f"Agreement:                {agree}/{len(rows)}")
    print(f"Disagreements:            {len(disagreements)}")
    if errors:
        print(f"Judge errors:             {len(errors)} (model unreachable?)")
    print(f"Time:                     {secs:.0f}s")
    print("=" * 54)

    # Report
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = []
    md.append(f"# Recall judged report - {when}")
    md.append("")
    src_name = Path(src).name
    judge_model = args.judge_model or J.config.JUDGE_MODEL
    md.append(f"Source answers: `{src_name}`  |  Judge: {judge_model}")
    md.append("")
    md.append("## Substring scorer vs LLM judge")
    md.append("")
    md.append(f"- Substring pass rate: **{sub_correct}/{len(rows)}**")
    md.append(f"- Judge pass rate: **{judge_correct}/{len(rows)}**")
    md.append(f"- They agree on **{agree}/{len(rows)}**; they disagree on **{len(disagreements)}**")
    md.append("")
    md.append("| Question | Substring | Judge | Judge reason |")
    md.append("|---|---|---|---|")
    for x in rows:
        flag = "" if x["agree"] else " ⚠"
        md.append(f"| {x['id']} | {x['substring']} | **{x['judge']}**{flag} | {x['judge_reason']} |")
    md.append("")
    if disagreements:
        md.append("## Disagreements (where the judge earns or loses trust)")
        md.append("")
        for x in disagreements:
            md.append(f"### {x['id']} - {x['question']}")
            md.append(f"- Requirement: {x['requirement']}")
            md.append(f"- Substring said **{x['substring']}**, judge said **{x['judge']}** - {x['judge_reason']}")
            ans = (x["answer"] or "").strip().replace("\n", " ")
            if len(ans) > 260:
                ans = ans[:260] + "..."
            md.append(f"- Answer: {ans}")
            md.append("")

    REPORTS_DIR.mkdir(exist_ok=True)
    md_path = REPORTS_DIR / f"{stamp}_judged.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    json_path = REPORTS_DIR / f"{stamp}_judged.json"
    json_path.write_text(json.dumps({
        "source": Path(src).name,
        "judge_model": args.judge_model or J.config.JUDGE_MODEL,
        "substring_pass": sub_correct,
        "judge_pass": judge_correct,
        "agreement": agree,
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nReport: {md_path}")
    print(f"Data:   {json_path}")


if __name__ == "__main__":
    main()
