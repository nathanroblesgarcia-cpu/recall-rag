"""
agent.py - The agent: one tool-calling loop over BOTH your notes and live numbers.

What it does:
  1. You ask a question (e.g. "how much is in the emergency fund now, and what is
     my target?").
  2. The local AI model (qwen2.5:7b) decides what it needs: a number, a fact from
     your notes, or both.
  3. It calls the tools: search_notes (Recall's hybrid note search) and/or the
     number tools in cafe_tools.py (FICTIONAL café numbers in this demo; in real
     use they would read your own systems, read-only).
  4. The tool results are handed back to the model.
  5. The model writes a plain-language answer from the real numbers and notes.

Guards (each added because the agent eval caught the failure it prevents):
  - answering without looking anything up  -> sent back once to look first;
                                              if it still won't, the note search
                                              is run for it
  - talking about "the notes" without searching them -> sent back to search
  - a tool call WRITTEN as text             -> parsed and run as a real call
  - numbers/dates re-worded ("2040" -> "the 2040s") -> exact-copy rule

Everything runs on your machine. Nothing is sent to the cloud.

Run it:   python agent.py
Or ask a one-off:   python agent.py "What was the café's profit in August 2026?"
The web page uses it too: Mode "Notes + live numbers".
"""

import datetime
import json
import re
import requests

import cafe_tools
import config
import tracing

# The model that runs the tool-calling loop. qwen2.5:7b handles tools reliably;
# llama3.2:3b is faster but flakier at deciding tool calls, so we use the 7B here.
AGENT_MODEL = "qwen2.5:7b"

# How many think -> call-tool -> think rounds we allow before forcing an answer.
# A single question needs 1-2. The cap stops any runaway loop.
MAX_STEPS = 4


# ---------------------------------------------------------------------------
# 1. TOOL DEFINITIONS
# These describe each tool to the model in the format Ollama expects. The model
# reads these descriptions to decide which tool to call and what to pass.
# ---------------------------------------------------------------------------
_MONTH_ARG = {
    "month": {
        "type": "string",
        "description": "Month as YYYY-MM, e.g. '2026-08'. Omit for the latest month with data.",
    }
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search the owner's own written notes (plans, goals, targets, suppliers, team, recipes, standards, incidents, app settings). Use for anything that is written down rather than a live sales, cost, profit or balance figure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, in a few plain words, e.g. 'emergency fund target' or 'Verde minimum order'.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cafe_sales",
            "description": "Café sales for one month: total, by category (coffee drinks, retail beans, pastries), and drinks sold.",
            "parameters": {"type": "object", "properties": _MONTH_ARG, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cafe_expenses",
            "description": "Café costs for one month. Give a 'category' (rent, wages, green beans, milk and pastries, utilities) for just that line; omit it for all costs.",
            "parameters": {
                "type": "object",
                "properties": {
                    **_MONTH_ARG,
                    "category": {
                        "type": "string",
                        "description": "Optional cost category, e.g. 'green beans'. Omit for all costs.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cafe_profit",
            "description": "Café profit for one month: sales minus costs.",
            "parameters": {"type": "object", "properties": _MONTH_ARG, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cafe_cash_position",
            "description": "Current balances: operating account, emergency fund, expansion fund, and total. No arguments.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

NOTE_HITS = 5          # how many note pieces search_notes hands back
NOTE_CHARS = 700       # how much of each piece, so the desk stays small


def search_notes(query=""):
    """Recall's hybrid note search, as a tool. Returns the top note pieces with
    their source so the model can quote and cite them. rag is imported here, not
    at the top, so number-only questions never pay for loading the search model."""
    import rag
    query = (query or "").strip()
    if not query:
        return {"error": "empty query"}
    hits = rag.retrieve(query, top_k=NOTE_HITS)
    return {
        "query": query,
        "notes": [{"source": h["source"], "text": h["text"][:NOTE_CHARS]} for h in hits],
    }


def _run_tool(fn, args):
    if fn not in DISPATCH:
        return {"error": f"unknown tool {fn}"}
    return DISPATCH[fn](**args)


def _tool_summary(fn, result):
    """What is worth seeing in a trace from one tool's result: for note search,
    which notes came back; for number tools, a short preview of the numbers."""
    if isinstance(result, dict) and result.get("error"):
        return {"error": result["error"]}
    if fn == "search_notes" and isinstance(result, dict):
        return {"found": [n["source"] for n in result.get("notes", [])]}
    return {"result": tracing._short(result)}


# Maps a tool name (as the model calls it) to the real Python function.
DISPATCH = {
    "search_notes": search_notes,
    "cafe_sales": cafe_tools.cafe_sales,
    "cafe_expenses": cafe_tools.cafe_expenses,
    "cafe_profit": cafe_tools.cafe_profit,
    "cafe_cash_position": cafe_tools.cafe_cash_position,
}


def _system_prompt():
    """Tell the model who it is, the date (for 'last month'), and how to behave."""
    today = datetime.date.today()
    this_month = today.strftime("%Y-%m")
    last_month = (today.replace(day=1) - datetime.timedelta(days=1)).strftime("%Y-%m")
    return (
        f"You are {config.USER_FIRST_NAME}'s assistant for the café. You can look up "
        "the café's numbers (sales, costs, profit, cash and fund balances) and search "
        f"{config.USER_FIRST_NAME}'s own written notes (search_notes), by calling the "
        "provided tools. "
        f"Today is {today:%Y-%m-%d}. 'This month' is {this_month}. "
        f"'Last month' is {last_month}. "
        "For ACTUAL figures (sales, costs, profit, how much is in an account or fund), "
        "CALL A NUMBER TOOL. For anything written down (goals, targets, plans, "
        "suppliers, team, recipes, standards, why something happened), CALL "
        "search_notes. If a question needs both, call both. 'I', 'me', 'my' mean "
        f"{config.USER_FIRST_NAME}. All amounts are US dollars ($). "
        "Answer in two or three short, plain sentences. When you use notes, name "
        "the note you used, exactly as search_notes returned it. Never say you "
        "searched, or describe a note, unless search_notes actually returned it. "
        "If the notes returned do not answer the question, say so. "
        "Copy numbers, years and dates EXACTLY as the note or tool writes them; "
        "never round, re-word or change them (for example 'mid 2027' must stay "
        "'mid 2027', not 'the 2020s'). "
        "Do not show raw JSON. "
        "If a tool result contains an \"error\", or you did not get a number you "
        "need, say plainly that you could not get it. NEVER make up or guess a "
        "number that a tool did not return."
    )


def _chat(messages, use_tools=True, trace=None, round_no=None):
    """One call to the local Ollama chat API. Returns the assistant message dict.
    With a trace, the round is recorded: time, tokens in/out, what it decided."""
    if trace is None:
        return _chat_raw(messages, use_tools)[0]
    with trace.step("model", round=round_no, tools_offered=use_tools) as rec:
        msg, usage = _chat_raw(messages, use_tools)
        rec.update(usage)
        calls = msg.get("tool_calls") or []
        if calls:
            rec["decided"] = "call " + ", ".join(c["function"]["name"] for c in calls)
        else:
            rec["decided"] = "write answer"
            rec["text"] = tracing._short(msg.get("content") or "", 200)
    return msg


def _chat_raw(messages, use_tools=True):
    """The actual HTTP call. Returns (message dict, token usage dict)."""
    payload = {
        "model": AGENT_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": config.OLLAMA_NUM_CTX, "temperature": 0.1},
    }
    if use_tools:
        payload["tools"] = TOOL_SCHEMAS
    resp = requests.post(config.OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    body = resp.json()
    return body["message"], tracing.usage_from(body)


# Phrases that mean "I looked in the notes" or "it's not there" (see GUARD 3).
_NOTES_CLAIM = re.compile(
    r"\bnotes?\b|not mentioned|could not find|couldn't find|no information|"
    r"not (?:stated|specified|recorded|available)",
    re.IGNORECASE,
)


def _text_tool_call(content):
    """Small local models sometimes WRITE a tool call as text instead of making
    one, e.g. 'search_notes {"query": "Verde decaf process"}' or
    '{"name": "search_notes", "arguments": {...}}'. Left alone, the loop takes
    that text as the final answer (the eval caught exactly this). If the whole
    reply is one such call to a known tool, return it in the real tool_calls
    shape; otherwise return []."""
    text = (content or "").strip().strip("`").strip()
    if text.startswith("json"):
        text = text[4:].strip()
    name, args = None, None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and obj.get("name") in DISPATCH:
            name, args = obj["name"], obj.get("arguments") or obj.get("parameters") or {}
    except ValueError:
        # name {json} | name({json}) | name() | name   (whole reply, nothing else)
        m = re.fullmatch(r"(\w+)\s*(?:\(\s*(\{.*\})?\s*\)|(\{.*\}))?", text, re.S)
        if m and m.group(1) in DISPATCH:
            body = m.group(2) or m.group(3)
            try:
                args = json.loads(body) if body else {}
                name = m.group(1)
            except ValueError:
                pass
    if name is None or not isinstance(args, dict):
        return []
    return [{"function": {"name": name, "arguments": args}}]


def ask(question, verbose=True, on_step=None, trace=None):
    """
    Answer one question, using tools when needed.

    Returns a dict: {"answer": str, "tools_used": [ {name, args, result}, ... ]}
    verbose=True prints each step so you can watch what the AI decides to do.
    on_step, if given, is called with a small dict as each step happens, so the
    web page can show the agent working live:
      {"type": "thinking"}                      a model round is starting
      {"type": "tool", "name", "args"}          about to run a tool
      {"type": "notes", "notes": [...]}         what search_notes returned
      {"type": "nudge"}                         it tried to answer without looking
    trace, if given (a tracing.Trace), records every round, tool and guard.
    """
    def emit(ev):
        if on_step:
            try:
                on_step(ev)
            except Exception:
                pass  # a display hiccup must never break the answer
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": question},
    ]
    tools_used = []
    notes_nudged = False

    for step in range(MAX_STEPS):
        emit({"type": "thinking"})
        msg = _chat(messages, use_tools=True, trace=trace, round_no=step + 1)
        calls = msg.get("tool_calls") or []
        if not calls:
            calls = _text_tool_call(msg.get("content"))
            if calls:
                if verbose:
                    print("  -> tool call arrived as plain text; running it for real")
                if trace:
                    trace.note("guard", which="tool call written as text, run for real")
                msg = {"role": "assistant", "content": "", "tool_calls": calls}

        # GUARD: answering with no tool at all means answering from the model's
        # own memory, which is where made-up notes come from (the eval caught it
        # inventing a note title for a question it never searched). Every
        # question here is about his notes or his numbers, so if the first round
        # skips the tools, send it back once to look first.
        if not calls and not tools_used and step == 0:
            if verbose:
                print("  -> no tool called; asking it to look first")
            emit({"type": "nudge"})
            if trace:
                trace.note("guard", which="answered without looking, sent back to search")
            messages.append({"role": "assistant", "content": msg.get("content") or ""})
            messages.append({"role": "user", "content": (
                "You have not looked anything up yet, so you do not know the answer. "
                "Call search_notes (or a number tool if it is about actual figures) "
                "first, then answer only from what it returns."
            )})
            continue

        # GUARD 2 (the eval caught the nudge alone not being enough): if it
        # STILL will not use a tool, do the note search for it and put the
        # results on the desk, so it can only answer from what was found.
        if not calls and not tools_used:
            if verbose:
                print("  -> still no tool; running search_notes for it")
            if trace:
                trace.note("guard", which="still no tool, ran search_notes for it")
            calls = [{"function": {"name": "search_notes", "arguments": {"query": question}}}]
            msg = {"role": "assistant", "content": "", "tool_calls": calls}

        # GUARD 3: an answer that talks about "the notes" (or says it could not
        # find something) when search_notes was never called is a made-up
        # search. The eval caught "not mentioned in the notes" after only a
        # number tool ran. Send it back once to actually search.
        if (not calls and not notes_nudged
                and not any(t["name"] == "search_notes" for t in tools_used)
                and _NOTES_CLAIM.search(msg.get("content") or "")):
            if verbose:
                print("  -> talked about notes without searching; asking it to search")
            notes_nudged = True
            emit({"type": "nudge"})
            if trace:
                trace.note("guard", which="talked about notes without searching, sent back")
            messages.append({"role": "assistant", "content": msg.get("content") or ""})
            messages.append({"role": "user", "content": (
                "You mentioned the notes but have not searched them. Call "
                "search_notes for the part of the question that is written down. "
                "Then answer the WHOLE question: keep every figure a tool already "
                "gave you, and add what the notes say."
            )})
            continue

        # No tool call means the model is ready to answer in plain text.
        if not calls:
            answer = (msg.get("content") or "").strip()
            if verbose:
                print(f"\n[final answer after {len(tools_used)} tool call(s) in {step} round(s)]")
            return {"answer": answer, "tools_used": tools_used}

        # The model asked for one or more tools. Record its message, run each
        # tool, and feed the results back for the next round.
        messages.append(msg)
        for call in calls:
            fn = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if verbose:
                print(f"  -> tool: {fn}({args})")
            emit({"type": "tool", "name": fn, "args": args})
            rec = {}
            try:
                if trace:
                    with trace.step("tool", name=fn, args=args) as rec:
                        result = _run_tool(fn, args)
                else:
                    result = _run_tool(fn, args)
            except Exception as e:
                result = {"error": str(e)}
            if trace:
                rec.update(_tool_summary(fn, result))
            tools_used.append({"name": fn, "args": args, "result": result})
            if fn == "search_notes" and isinstance(result, dict) and result.get("notes"):
                emit({"type": "notes", "notes": result["notes"]})
            messages.append({"role": "tool", "content": json.dumps(result)})

    # Ran out of steps: ask once more without tools to force a written answer.
    emit({"type": "thinking"})
    if trace:
        trace.note("guard", which=f"hit the {MAX_STEPS}-round limit, forced an answer")
    msg = _chat(messages, use_tools=False, trace=trace, round_no="final")
    return {"answer": (msg.get("content") or "").strip(), "tools_used": tools_used}


# ---------------------------------------------------------------------------
# Command-line test.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        # Default demo questions if you don't pass one.
        questions = [
            "What was the café's profit in August 2026?",
            "How much is in the emergency fund now, and what is my target?",
        ]

    for q in questions:
        print("=" * 64)
        print("Q:", q)
        out = ask(q, verbose=True)
        print("\nA:", out["answer"])
        print(f"(used {len(out['tools_used'])} tool call(s))")
