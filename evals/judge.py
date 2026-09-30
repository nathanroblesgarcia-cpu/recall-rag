# judge.py
# LLM-as-judge: grade a Recall answer by MEANING, not by substring match.
#
# The substring scorer in run_eval.py can only ask "does this exact text appear?"
# so it marks a correct answer wrong when the wording, format, or rounding differs
# ("sixty-five degrees" vs "65"; "18 g in / 36 g out" vs "18/36"; a worded recipe
# vs the raw numbers). This judge fixes that: it reads the question,
# the required facts, and the actual answer, and decides correct / partial /
# incorrect regardless of phrasing.
#
# It runs on the LOCAL 7B model (config.JUDGE_MODEL) - free, private, offline,
# like the rest of Recall. Grading emits only a short verdict, so the slower model
# is affordable here, and its judgement is far more reliable than the 3B's.
#
# A judge is only useful if you can trust it, so it is deliberately strict: a
# missing fact, a wrong fact, or an "I couldn't find it" answer all fail, and a
# confident guess that the notes don't support is not rewarded. rejudge.py then
# shows every case where the judge and the substring scorer DISAGREE - those
# disagreements are where you inspect whether the judge earned its keep.

import json
import os
import re
import time

import requests
from dotenv import load_dotenv

import config

load_dotenv()  # so GEMINI_API_KEY is available for the optional Gemini judge

VALID = ("correct", "partial", "incorrect")

SYSTEM = (
    "You are a strict grader for a question-answering system. You are given a "
    "QUESTION, a REQUIREMENT listing the facts a correct answer must convey, and an "
    "ANSWER produced by the system. Decide whether the ANSWER satisfies the "
    "REQUIREMENT.\n"
    "Rules:\n"
    "- Wording, format, rounding, units, and extra detail DO NOT matter. Judge meaning.\n"
    "- Grade ONLY against the listed REQUIREMENT. Never invent extra criteria (units, "
    "currency, precision, source citations) that the requirement does not ask for.\n"
    "- Extra correct detail, and any units or currency the answer volunteers, are fine "
    "and never a reason to lower the verdict.\n"
    "- A 'one of (A / B / C)' group is satisfied if ANY listed value appears in the "
    "ANSWER, in any wording or format. Check the answer against EVERY value in the "
    "group, not just the first one; a match on any single listed value passes the "
    "group. Extra numbers the answer adds (totals, differences, percentages) are "
    "irrelevant and NEVER make a present value 'not match'.\n"
    "- Read the whole ANSWER for the fact. Only mark it a non-answer if it literally says "
    "it could not find or does not have the information.\n"
    "- 'correct' = every required fact is present and right.\n"
    "- 'partial' = at least one required fact is present and right, but not all.\n"
    "- 'incorrect' = a required fact is missing or wrong, OR the answer says it could "
    "not find the information, OR it states a fact the requirement contradicts.\n"
    "- Do NOT reward a confident guess that the requirement does not support.\n"
    'Respond with ONLY compact JSON, no prose: '
    '{"verdict":"correct|partial|incorrect","reason":"<=20 words"}'
)

# Worked examples (few-shot). Each teaches one rule the judge tends to get wrong.
# They are generic on purpose - they teach the principle, they don't leak the real
# quiz answers. The three "correct" cases guard the false negatives seen in Run 3
# (extra detail, volunteered currency, a fact stated inside prose); the "incorrect"
# and "partial" cases stop the fix from sliding into rubber-stamping.
FEWSHOT = [
    # reworded / different format -> still correct
    {"q": "What is the storage limit?",
     "req": "one of (2 TB / 2,000 GB)",
     "ans": "The limit is two terabytes.",
     "verdict": "correct", "reason": "two terabytes equals 2 TB"},
    # required fact present, buried in extra detail -> correct (do not dock detail)
    {"q": "What year does the rollout finish?",
     "req": "one of (2027)",
     "ans": "Given the current pace and a few assumptions, it should wrap up around late 2027.",
     "verdict": "correct", "reason": "states 2027; extra detail is fine"},
    # answer volunteers a currency that was not required -> correct (do not invent rules)
    {"q": "What is the monthly fee?",
     "req": "one of (500 / 500.00)",
     "ans": "The monthly fee is USD 500 per user.",
     "verdict": "correct", "reason": "required figure present; currency not required"},
    # required value is NOT first in its group, and is surrounded by distractor
    # numbers -> still correct. Teaches: scan every option in a 'one of' group, and
    # ignore extra totals/percentages the answer volunteers.
    {"q": "How much Ethiopian do we order monthly, and what is the lead time?",
     "req": "one of (240 / 240 kg) AND one of (2 weeks / two weeks / 14 days)",
     "ans": "We order 240 kg a month and the lead time is about two weeks, so I reorder before dropping below a month of stock.",
     "verdict": "correct", "reason": "240 and two weeks are listed options and both appear"},
    # fact stated inside prose -> correct (read the whole answer)
    {"q": "Why did the espresso machine go down?",
     "req": "one of (breaker / circuit) AND one of (overload / second grinder)",
     "ans": "A tripped breaker: the second grinder overloaded the circuit they shared.",
     "verdict": "correct", "reason": "names the breaker and the overload cause"},
    # genuine refusal -> incorrect (do not rescue)
    {"q": "What is the licence key?",
     "req": "one of (the key value)",
     "ans": "I couldn't find any information about the licence key in the notes.",
     "verdict": "incorrect", "reason": "answer could not find the information"},
    # confident wrong fact -> incorrect
    {"q": "Which database does the app use?",
     "req": "one of (PostgreSQL / postgres)",
     "ans": "The app uses MongoDB.",
     "verdict": "incorrect", "reason": "names the wrong database"},
    # some but not all required facts -> partial
    {"q": "Give the size before and after.",
     "req": "one of (100 MB) AND one of (40 MB)",
     "ans": "It shrank to 40 MB.",
     "verdict": "partial", "reason": "after present, before missing"},
]


def _fewshot_messages():
    """Turn FEWSHOT into alternating user/assistant turns the model learns from."""
    msgs = []
    for ex in FEWSHOT:
        msgs.append({"role": "user", "content": (
            f"QUESTION: {ex['q']}\n\nREQUIREMENT (the answer must convey): {ex['req']}\n\n"
            f"ANSWER: {ex['ans']}\n\nGrade the ANSWER now."
        )})
        msgs.append({"role": "assistant", "content": json.dumps(
            {"verdict": ex["verdict"], "reason": ex["reason"]}, separators=(",", ":"))})
    return msgs


def requirement_of(item):
    """Turn an eval item's expected_facts into one readable requirement line for the
    judge. A list of options becomes 'one of: a / b / c'; separate facts are joined
    with 'AND' so the judge knows all are required."""
    parts = []
    for f in item["expected_facts"]:
        if isinstance(f, list):
            parts.append("one of (" + " / ".join(str(x) for x in f) + ")")
        else:
            parts.append(str(f))
    return " AND ".join(parts) if parts else "(no specific fact required)"


def _parse(text):
    """Local models don't always return clean JSON. Pull the first {...} block and
    read it; if that fails, sniff for a verdict word. Always returns a valid dict."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            v = str(d.get("verdict", "")).strip().lower()
            if v in VALID:
                return {"verdict": v, "reason": str(d.get("reason", ""))[:160]}
        except Exception:
            pass
    t = (text or "").lower()
    if "incorrect" in t:
        return {"verdict": "incorrect", "reason": (text or "")[:160]}
    if "partial" in t:
        return {"verdict": "partial", "reason": (text or "")[:160]}
    if "correct" in t:
        return {"verdict": "correct", "reason": (text or "")[:160]}
    return {"verdict": "incorrect", "reason": "unparseable judge reply: " + (text or "")[:120]}


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _ask_gemini(messages, model):
    """Send the same judge conversation to Google Gemini instead of the local model.

    Why a cloud judge at all? A local 7B grading a local 3B is two small models
    agreeing with each other. A much larger model from a different family is an
    independent second opinion: where it and the local judge disagree is exactly
    where a human should look. The quiz is fictional sample data, so sending it
    out is harmless. Needs GEMINI_API_KEY in your .env (free key from Google AI
    Studio). Uses plain requests, no extra package.
    """
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in your .env")
    # Gemini takes the system prompt separately and calls the assistant "model".
    system = next(m["content"] for m in messages if m["role"] == "system")
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages if m["role"] != "system"
    ]
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    # The key goes in a header, never the URL, so it can't end up in logs.
    headers = {"x-goog-api-key": key}
    # Free-tier Gemini answers 429 (rate limit) or 503 (busy). A per-minute limit
    # or a busy model clears in seconds, so wait and retry. A per-DAY limit (the
    # free tier allows only a handful of requests per model per day) will not
    # clear until tomorrow, so stop at once instead of retrying for minutes.
    for attempt in range(5):
        resp = requests.post(GEMINI_URL.format(model=model), json=body, headers=headers, timeout=120)
        if resp.status_code == 429 and "PerDay" in resp.text:
            raise RuntimeError(f"Gemini free-tier DAILY limit reached for {model}; try another model or tomorrow")
        if resp.status_code not in (429, 503) or attempt == 4:
            break
        time.sleep(5 * 2 ** attempt)
    resp.raise_for_status()
    parts = resp.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def judge(question, requirement, answer, model=None):
    """Grade one answer. Returns {"verdict", "reason"}. verdict is one of
    correct/partial/incorrect, or "error" if the judge model was unreachable.

    model: an Ollama model name (local), or any name starting with "gemini"
    (e.g. gemini-flash-latest) to grade with Google Gemini instead."""
    if not answer:
        return {"verdict": "incorrect", "reason": "no answer produced"}
    user = (
        f"QUESTION: {question}\n\n"
        f"REQUIREMENT (the answer must convey): {requirement}\n\n"
        f"ANSWER: {answer}\n\n"
        "Grade the ANSWER now."
    )
    messages = [{"role": "system", "content": SYSTEM}]
    messages.extend(_fewshot_messages())
    messages.append({"role": "user", "content": user})
    model = model or config.JUDGE_MODEL
    if model.startswith("gemini"):
        try:
            return _parse(_ask_gemini(messages, model))
        except Exception as e:
            return {"verdict": "error", "reason": f"judge unavailable: {e}"}
    try:
        resp = requests.post(
            config.OLLAMA_URL,
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                # Temperature 0: grading should be as repeatable as the model allows.
                "options": {"temperature": 0, "num_ctx": config.OLLAMA_NUM_CTX},
            },
            timeout=240,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
    except Exception as e:
        return {"verdict": "error", "reason": f"judge unavailable: {e}"}
    return _parse(content)
