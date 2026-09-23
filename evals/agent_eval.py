# agent_eval.py
# The capstone quiz: grades agent.py, the ONE agent that has both note search
# (search_notes) and the live number tools (cafe_tools.py: fictional café numbers in this demo).
#
# Three checks per question:
#   1. ROUTING   Did it call every tool the question needs?          (objective)
#   2. FACTS     Does the answer contain the known note facts, and none of the
#                known-wrong ones (forbidden_facts)?                   (substring)
#   3. GROUNDED  Does the answer quote a number a tool really returned? (anti-made-up)
#
# Money numbers change daily, so GROUNDED compares the answer against the tool's
# own result from that run, never a hard-coded value.
#
# USAGE (from the Recall folder, venv active; needs Ollama running):
#     python evals\agent_eval.py
#     python evals\agent_eval.py --limit 3

import argparse
import contextlib
import io
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent      # noqa: E402
from run_eval import fact_present  # noqa: E402  (same fact format as golden.json)

GOLDEN_PATH = Path(__file__).resolve().parent / "agent_golden.json"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def _numbers_in(obj, out):
    """Collect every number in a tool result (any depth)."""
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers_in(v, out)
    return out


def _answer_numbers(answer):
    """Every number written in the answer, commas removed ('$14,400.00' -> 14400.0)."""
    return [float(m.replace(",", "")) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", answer or "")]


def grounded(answer, tools_used):
    """True if the answer quotes at least one number a NUMBER tool returned,
    allowing rounding (within 1%, or within 1 for small values) and a dropped
    minus sign ('a loss of $500' for -500). Note-search results are ignored: they
    hold the notes' numbers, not live ones."""
    tool_nums = []
    for t in tools_used:
        if t["name"] != "search_notes":
            _numbers_in(t["result"], tool_nums)
    tool_nums = [abs(n) for n in tool_nums if abs(n) >= 1]
    for a in _answer_numbers(answer):
        for n in tool_nums:
            if abs(a - n) <= max(1.0, 0.01 * n):
                return True
    return False


def run(items):
    results = []
    for n, item in enumerate(items, start=1):
        q = item["question"]
        print(f"[{n}/{len(items)}] {item['id']}: {q}", flush=True)
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            out = agent.ask(q, verbose=False)
        secs = time.time() - t0
        used = [t["name"] for t in out["tools_used"]]
        routed = all(t in used for t in item["expected_tools"])
        missing = [f for f in item["expected_facts"] if not fact_present(out["answer"], f)]
        missing += [f"NOT {f}" for f in item.get("forbidden_facts", []) if fact_present(out["answer"], f)]
        ground = grounded(out["answer"], out["tools_used"]) if item["grounded_number"] else None
        passed = routed and not missing and (ground is not False)
        results.append({
            "id": item["id"], "kind": item["kind"], "question": q,
            "expected_tools": item["expected_tools"], "tools_used": used,
            "tool_args": [t["args"] for t in out["tools_used"]],
            "routed": routed, "facts_missing": missing, "grounded": ground,
            "passed": passed, "answer": out["answer"], "seconds": round(secs, 1),
        })
        print(f"    tools={used} routed={routed} facts_ok={not missing} grounded={ground} "
              f"-> {'PASS' if passed else 'FAIL'} ({secs:.0f}s)", flush=True)
    return results


def render(results, when):
    total = len(results)
    lines = [f"# Recall agent eval - {when}", "",
             f"Model: {agent.AGENT_MODEL}  |  Questions: {total}", "", "## Headline", ""]
    lines.append(f"- Overall pass: **{sum(r['passed'] for r in results)}/{total}**")
    lines.append(f"- Routing (called the right tools): **{sum(r['routed'] for r in results)}/{total}**")
    g = [r for r in results if r["grounded"] is not None]
    lines.append(f"- Grounded numbers: **{sum(bool(r['grounded']) for r in g)}/{len(g)}**")
    for kind in ("notes", "numbers", "mixed"):
        k = [r for r in results if r["kind"] == kind]
        if k:
            lines.append(f"- {kind}: {sum(r['passed'] for r in k)}/{len(k)}")
    lines += ["", "## Per question", "",
              "| Question | Kind | Tools used | Routed | Facts | Grounded | Result | Secs |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        facts = "ok" if not r["facts_missing"] else "MISSING"
        gr = "-" if r["grounded"] is None else ("yes" if r["grounded"] else "NO")
        lines.append(f"| {r['id']} | {r['kind']} | {', '.join(r['tools_used']) or '(none)'} | "
                     f"{'yes' if r['routed'] else 'NO'} | {facts} | {gr} | "
                     f"{'PASS' if r['passed'] else 'FAIL'} | {r['seconds']} |")
    lines += ["", "## Answers", ""]
    for r in results:
        lines.append(f"**{r['id']}** - {r['answer'].strip()}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Run the Recall agent (capstone) quiz.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    items = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["items"]
    if args.limit:
        items = items[:args.limit]
    results = run(items)

    total = len(results)
    print("\n" + "=" * 50)
    print(f"Overall pass: {sum(r['passed'] for r in results)}/{total}")
    print(f"Routing:      {sum(r['routed'] for r in results)}/{total}")
    g = [r for r in results if r["grounded"] is not None]
    print(f"Grounded:     {sum(bool(r['grounded']) for r in g)}/{len(g)}")
    print("=" * 50)

    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_path = REPORTS_DIR / f"{stamp}_agent.md"
    md_path.write_text(render(results, when), encoding="utf-8")
    (REPORTS_DIR / f"{stamp}_agent.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {md_path}")


if __name__ == "__main__":
    main()
