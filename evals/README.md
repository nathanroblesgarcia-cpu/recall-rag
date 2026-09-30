# Evals

Why this folder exists: anyone can build a RAG demo that looks good on one lucky
question. The difference between a demo and something you would ship is being able
to **measure** whether it actually answers correctly, and catch it when a change
makes it worse. That is what these evals do.

## What gets measured

Each question in `golden.json` is scored on two independent things:

1. **Retrieval** — did the right note get pulled into the sources? This is the
   core RAG metric and needs no answer model, so it runs in seconds. Reported
   three ways: **hit rate** (found at all), **top 3** (found in the first three),
   and **MRR** (mean reciprocal rank: 1.0 means the right note is always first).
   Hit rate alone hides rank problems; a note found at rank 6 "passes" but may
   never reach the answer model.
2. **Answer** — did the written answer actually contain the known facts?

## The files

| File | What it does |
|---|---|
| `golden.json` | the quiz: questions whose answers are known from the sample notes |
| `run_eval.py` | asks each question, scores retrieval and answer, writes a report |
| `judge.py` | an LLM-as-judge pass: a stronger local model grades each answer |
| `rejudge.py` | compares the judge against the plain fact-check, and optionally against a second judge (`--compare`) |
| `judge_calibration.json` | hand-written answers with known grades, planted mistakes, to test the graders |
| `judge_calibration.py` | scores each grader (substring, local judge, Gemini) against those known grades |
| `agent_golden.json` | the agent quiz: notes-only, numbers-only and mixed questions |
| `agent_eval.py` | grades `agent.py`: tool routing, note facts, grounded numbers |
| `reports/` | timestamped run history (gitignored) |

## Run it

```bash
python run_eval.py                  # retrieval + answer
python run_eval.py --retrieval-only # fast: retrieval only, no answer model
python run_eval.py --limit 5        # first 5 questions
python judge.py                     # LLM-as-judge grading
python rejudge.py --compare gemini-flash-latest  # judge the judge (needs GEMINI_API_KEY)
python judge_calibration.py --judges gemini-flash-lite-latest  # grade the graders
python agent_eval.py                # the agent quiz
```

## The agent quiz

`agent_eval.py` checks each answer from `agent.py` three ways:

1. **Routing** — did it call every tool the question needs (notes, numbers, or both)?
2. **Facts** — does the answer contain the known facts, and none of the listed
   `forbidden_facts`? The forbidden list exists because a substring check passes
   near-misses: "2040" is inside "the 2040s", a whole decade instead of one year.
3. **Grounded numbers** — does the answer quote a figure a number tool actually
   returned in that run? Numbers are checked against the tool's own output, never
   hard-coded, so this catches made-up figures.

Building the agent against this quiz is what surfaced its guards: it answered
without looking anything up (and once described a note that did not exist), it
wrote a tool call as plain text instead of making it, and it re-worded dates.
Each is now caught in code, not just discouraged in the prompt.

## Why two scorers (substring and LLM-as-judge)

The plain scorer checks whether the required fact string is present. It is fast
and objective but strict: "sixty-five degrees" would fail a check for "65"
even though it is correct. The **LLM-as-judge** reads the question, the required
fact, and the answer, and gives a verdict, so it accepts correct answers phrased
differently. Running both and looking at where they disagree (`rejudge.py`) is how
you find both real misses and scorer blind spots.

### A second, independent judge (Gemini)

A local 7B judge grading a local 3B answer model is two small models checking each
other. `judge.py` can also grade with Google Gemini (any model name starting with
`gemini`), a much larger model from a different family, and `rejudge.py --compare`
runs both judges on the same saved answers and lists where they split. A split is
where a human should read the answer, because one of the judges is wrong.

On the café quiz, `gemini-flash-lite-latest` agreed with the fact-check on 24/24 in
75 seconds, where the local 7B needs minutes per answer on a busy CPU. That result
is a wiring check, not proof of judgement: every answer was right, so the judge
never had to catch a wrong one.

Free-tier notes learned the hard way: each Gemini model allows only a small number
of requests **per day** (20 for the default Flash at the time of writing), so a
24-question run can exhaust it. A per-minute limit is retried; a per-day limit
stops the call at once with a clear message instead of retrying for half an hour.
Flash-Lite has its own, separate daily allowance and is plenty for grading.

### Grading the graders: the calibration set

Agreement between two graders says nothing about which one is right.
`judge_calibration.json` fixes that: 16 hand-written answers with a **known**
correct grade, each planting one mistake. Some are wrong answers built to fool a
text match (the right digits inside a wrong number, "600" for "60"; keywords
mentioned only to deny them; three guessed years). Some are correct answers in
different words ("72 hours" for "3 days", "twenty-five thousand"). A few are plain
controls. `judge_calibration.py` scores every grader against those labels and
splits the misses into **false pass** (a wrong answer graded correct, which hides
a bug) and **false fail** (a correct answer graded wrong, which is noise).

| Grader | Right | False pass | False fail |
|---|---|---|---|
| substring fact-check | 7/16 | 4 | 5 |
| gemini-flash-lite-latest | 16/16 | 0 | 0 |

The substring check fell for every trap it was built to test, in both directions.
The Gemini judge caught all of them. A perfect score on 16 obvious traps is a
floor, not a ceiling: the next step is harder cases (answers that are mostly right
with one subtle error) until the judge starts to miss, because a test that nothing
fails cannot tell two judges apart.

## A result worth keeping: hybrid search on a tiny corpus

On this 16-note demo set, meaning-only and hybrid search both score 24/24 with
MRR 1.0: every question has one obvious note, so there is nothing for exact-word
search to rescue. A demo corpus proves the pipeline is wired correctly; it cannot prove a
retrieval upgrade helps. You need a corpus messy enough to fail.

## How to extend it

Add an item to `golden.json`: a `question`, the `expected_sources` (a substring of
the source note's label), and `expected_facts` (a string that must appear, or a
list where any one option counts). Re-run. That is the whole loop.
