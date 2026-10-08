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
`judge_calibration.json` fixes that: hand-written answers with a **known**
correct grade, each planting one mistake. Some are wrong answers built to fool a
text match (the right digits inside a wrong number, "600" for "60"; keywords
mentioned only to deny them; three guessed years). Some are correct answers in
different words ("72 hours" for "3 days", "twenty-five thousand"). A few are plain
controls. `judge_calibration.py` scores every grader against those labels and
splits the misses into **false pass** (a wrong answer graded correct, which hides
a bug) and **false fail** (a correct answer graded wrong, which is noise).

The set grew in three rounds, and each round taught something:

**Round 1, basic (16 obvious traps).** The substring check scored 7/16 and fell
for every trap in both directions. Gemini scored 16/16. A perfect score on easy
traps is a floor, not a ceiling: a test nothing fails cannot tell judges apart.

**Round 2, hard (12 answers that are mostly right with one subtle error).** Dose
and yield swapped ("36 g in, 18 g out"), "240 grams" for 240 kg, "65 Fahrenheit"
for 65 Celsius, "at most 3 days" for at least, the right name in the wrong role.
Gemini dropped to **6/12 and passed 5 wrong answers**. Its reasons gave the cause
away: "conveys the required 240". The judge only saw the bare required fact, and
its prompt said units do not matter, so it could not tell 240 kg from 240 g. The
model was not the weak link; the grading setup was.

The fix: each golden question now carries a full `reference` answer ("240 kg of
Ethiopian Yirgacheffe every month"), and the judge prompt says a change of
meaning (unit that changes the amount, swapped values, flipped qualifier, right
value on the wrong thing) is wrong even when the required text appears, while an
equivalent value in another unit is still right. Hard set: **12/12**.

**Round 3, holdout (8 new traps, written after the fix, run once, never tuned on).**
Round 2's fix was written while looking at round 2's misses, which is teaching to
the test. The holdout uses traps the prompt does not name: "1,200 a year" for a
month, the target reported as the current balance, two ports swapped, "no minimum"
when 60 kg is the minimum, "3 weeks" for 3 days. Gemini: **7/8, zero false
passes**. Its one miss was too strict, not too lenient: it called a right recipe
with a wrong shot time incorrect instead of partial. Left untuned on purpose.

| Grader | Basic | Hard | Holdout | Wrong answers passed (of 24) |
|---|---|---|---|---|
| substring fact-check | 7/16 | 2/12 | 2/8 | 18 |
| Gemini, bare requirement | 16/16 | 6/12 | not run | 5 so far |
| Gemini, with reference answer | 16/16 | 12/12 | 7/8 | **0** |
| local qwen2.5:7b, with reference answer | 15/16 | 10/12 | 6/8 | 2 |

Lesson: an LLM judge is only as good as what it is shown. Give it the full right
answer, not a keyword, and measure it on traps it was not tuned on.
`python judge_calibration.py --no-reference` reruns the bare-requirement version
for the A/B.

### Local judge vs cloud judge, and why voting did not help

The local 7B judge (free, offline, same prompt and reference answers) scored 31/36
with **2 false passes**: it called a half recipe ("18 g of coffee", yield missing)
fully correct, and accepted "the fund already holds 25,000" when 25,000 is the
target, not the balance. Its other misses were the safe kind (too strict).

A common fix for a shaky judge is to ask it several times and take a vote. We
tried it: each answer graded 5 times with some randomness (temperature 0.7),
compared with one grade at temperature 0.

| Rule | Right | False pass | False fail |
|---|---|---|---|
| one vote, temperature 0 | 31/36 | 2 | 1 |
| majority of 5 | 31/36 | 2 | 1 |
| strict: any doubt fails it | 31/36 | 1 | 2 |

Majority voting changed nothing, and the strict rule only swapped one error for
another, at five times the cost. The reason shows in the raw votes: the half
recipe was graded "correct" 5 times out of 5. These are **systematic blind
spots, not random noise**, and voting only cancels noise. Check whether a judge's
errors repeat before paying for votes.

How we use the two judges: the local model is the free, fast judge while
iterating on Recall; Gemini is the grade we quote.

## A result worth keeping: hybrid search on a tiny corpus

On this 16-note demo set, meaning-only and hybrid search both score 24/24 with
MRR 1.0: every question has one obvious note, so there is nothing for exact-word
search to rescue. A demo corpus proves the pipeline is wired correctly; it cannot prove a
retrieval upgrade helps. You need a corpus messy enough to fail.

## How to extend it

Add an item to `golden.json`: a `question`, the `expected_sources` (a substring of
the source note's label), and `expected_facts` (a string that must appear, or a
list where any one option counts). Re-run. That is the whole loop.
