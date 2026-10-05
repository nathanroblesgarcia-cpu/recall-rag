# judge_calibration.py
# Grade the GRADERS: measure how often each scorer gives the right verdict.
#
# rejudge.py can only show where two graders DISAGREE. It cannot say which one is
# right, because nobody knows the true grade of those answers. This script fixes
# that with judge_calibration.json: hand-written answers whose correct grade is
# known in advance, each planting one specific mistake ("the right digits inside
# a wrong number", "keywords mentioned only to deny them", "a correct answer in
# different words"). Every scorer is then graded against those known labels.
#
# The substring fact-check from run_eval.py is always included as the baseline,
# so you can see exactly which traps it falls for and whether a judge does better.
#
# USAGE (from the Recall folder, with its venv):
#     python evals\judge_calibration.py                                  # local judge
#     python evals\judge_calibration.py --judges gemini-flash-lite-latest
#     python evals\judge_calibration.py --judges qwen2.5:7b gemini-flash-lite-latest
#
# Writes a *_calibration.md and *_calibration.json to evals\reports\.

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
sys.path.insert(0, str(BASE_DIR))

import judge as J  # noqa: E402

CALIBRATION_PATH = BASE_DIR / "judge_calibration.json"
GOLDEN_PATH = BASE_DIR / "golden.json"
REPORTS_DIR = BASE_DIR / "reports"
SUBSTRING = "substring"


def _norm(s):
    return (s or "").lower()


def substring_verdict(answer, expected_facts):
    """The same fact-presence check run_eval.py uses, mapped onto the judge's
    three-way scale. Copied (not imported) so this runs without loading the RAG."""
    if not expected_facts:
        return "correct"
    hits = 0
    for f in expected_facts:
        opts = f if isinstance(f, list) else [f]
        if any(_norm(o) in _norm(answer) for o in opts):
            hits += 1
    if hits == len(expected_facts):
        return "correct"
    return "partial" if hits else "incorrect"


def load_items():
    golden = {g["id"]: g for g in json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["items"]}
    items = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))["items"]
    for it in items:
        g = golden[it["golden_id"]]
        it["question"] = g["question"]
        it["expected_facts"] = g["expected_facts"]
        it["reference"] = g.get("reference")
    return items


def main():
    ap = argparse.ArgumentParser(description="Measure each grader against known correct grades.")
    ap.add_argument("--judges", nargs="*", default=None,
                    help="Judge models to test (Ollama names, or gemini-*). Default: config.JUDGE_MODEL.")
    ap.add_argument("--no-reference", action="store_true",
                    help="Hide golden.json's full reference answer from the judges (bare requirement only), for an A/B.")
    args = ap.parse_args()
    judges = args.judges or [J.config.JUDGE_MODEL]

    items = load_items()
    print(f"{len(items)} calibration answers | graders: {SUBSTRING}, {', '.join(judges)} | "
          f"reference answers {'HIDDEN' if args.no_reference else 'shown'} to judges\n")

    rows = []
    t0 = time.time()
    for n, it in enumerate(items, start=1):
        req = J.requirement_of(it)
        verdicts = {SUBSTRING: {"verdict": substring_verdict(it["answer"], it["expected_facts"]), "reason": ""}}
        for m in judges:
            ref = None if args.no_reference else it.get("reference")
            verdicts[m] = J.judge(it["question"], req, it["answer"], model=m, reference=ref)
        marks = "  ".join(f"{g}={'ok' if v['verdict'] == it['label'] else v['verdict'].upper()}"
                          for g, v in verdicts.items())
        print(f"[{n}/{len(items)}] {it.get('set', 'basic'):<5} {it['id']:<22} label={it['label']:<9} {marks}")
        rows.append({**it, "requirement": req, "verdicts": verdicts})
    secs = time.time() - t0

    graders = [SUBSTRING] + judges
    sets = ["all"] + sorted({r.get("set", "basic") for r in rows})

    def tally(g, subset):
        right = sum(1 for r in subset if r["verdicts"][g]["verdict"] == r["label"])
        errors = sum(1 for r in subset if r["verdicts"][g]["verdict"] == "error")
        # The costly mistake for an eval is passing a wrong answer: it hides a bug.
        false_pass = sum(1 for r in subset if r["verdicts"][g]["verdict"] == "correct" and r["label"] != "correct")
        false_fail = sum(1 for r in subset if r["verdicts"][g]["verdict"] != "correct" and r["label"] == "correct")
        return {"right": right, "of": len(subset), "errors": errors, "false_pass": false_pass, "false_fail": false_fail}

    score = {g: {s: tally(g, [r for r in rows if s == "all" or r.get("set", "basic") == s]) for s in sets}
             for g in graders}

    print("\n" + "=" * 70)
    print(f"{'Grader':<28}{'Set':<7}{'Right':>8}{'False pass':>12}{'False fail':>12}")
    for g in graders:
        for s in sets:
            t = score[g][s]
            err = f"  ({t['errors']} errors)" if t["errors"] else ""
            print(f"{g if s == 'all' else '':<28}{s:<7}{t['right']:>5}/{t['of']:<2}{t['false_pass']:>12}{t['false_fail']:>12}{err}")
    print(f"Time: {secs:.0f}s")
    print("=" * 70)

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    md = [f"# Judge calibration - {datetime.now():%Y-%m-%d %H:%M}", "",
          f"Reference answers {'hidden from' if args.no_reference else 'shown to'} the judges.", ""]
    md.append("Each answer was written by hand with a known correct grade. "
              "**False pass** = a wrong answer graded correct (hides bugs, the costly one). "
              "**False fail** = a correct answer graded wrong (noise).")
    md.append("")
    md.append("| Grader | Set | Right | False pass | False fail |")
    md.append("|---|---|---|---|---|")
    for g in graders:
        for s in sets:
            t = score[g][s]
            md.append(f"| {g} | {s} | **{t['right']}/{t['of']}** | {t['false_pass']} | {t['false_fail']} |")
    md.append("")
    md.append("| Answer | Set | Trap | Label | " + " | ".join(graders) + " |")
    md.append("|---|---|---|---|" + "---|" * len(graders))
    for r in rows:
        cells = []
        for g in graders:
            v = r["verdicts"][g]["verdict"]
            cells.append(v if v == r["label"] else f"**{v}** ✗")
        md.append(f"| {r['id']} | {r.get('set', 'basic')} | {r['trap']} | {r['label']} | " + " | ".join(cells) + " |")
    md.append("")
    misses = [(r, g) for r in rows for g in judges if r["verdicts"][g]["verdict"] != r["label"]]
    if misses:
        md.append("## Judge mistakes")
        md.append("")
        for r, g in misses:
            md.append(f"- **{g}** on `{r['id']}` said **{r['verdicts'][g]['verdict']}**, "
                      f"should be **{r['label']}** ({r['why']}) - judge's reason: {r['verdicts'][g]['reason']}")
        md.append("")

    REPORTS_DIR.mkdir(exist_ok=True)
    md_path = REPORTS_DIR / f"{stamp}_calibration.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    json_path = REPORTS_DIR / f"{stamp}_calibration.json"
    json_path.write_text(json.dumps({"graders": graders, "score": score, "rows": rows},
                                    indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {md_path}")


if __name__ == "__main__":
    main()
