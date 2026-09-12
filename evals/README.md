# Evals

Why this folder exists: anyone can build a RAG demo that looks good on one lucky
question. The difference between a demo and something you would ship is being able
to **measure** whether it actually answers correctly, and catch it when a change
makes it worse. That is what these evals do.

## What gets measured

Each question in `golden.json` is scored on two independent things:

1. **Retrieval** — did the right note get pulled into the sources? This is the
   core RAG metric and needs no answer model, so it runs in seconds.
2. **Answer** — did the written answer actually contain the known facts?

## The files

| File | What it does |
|---|---|
| `golden.json` | the quiz: questions whose answers are known from the sample notes |
| `run_eval.py` | asks each question, scores retrieval and answer, writes a report |
| `judge.py` | an LLM-as-judge pass: a stronger local model grades each answer |
| `rejudge.py` | compares the judge against the plain fact-check to find disagreements |
| `reports/` | timestamped run history (gitignored) |

## Run it

```bash
python run_eval.py                  # retrieval + answer
python run_eval.py --retrieval-only # fast: retrieval only, no answer model
python run_eval.py --limit 5        # first 5 questions
python judge.py                     # LLM-as-judge grading
```

## Why two scorers (substring and LLM-as-judge)

The plain scorer checks whether the required fact string is present. It is fast
and objective but strict: "sixty-five degrees" would fail a check for "65"
even though it is correct. The **LLM-as-judge** reads the question, the required
fact, and the answer, and gives a verdict, so it accepts correct answers phrased
differently. Running both and looking at where they disagree (`rejudge.py`) is how
you find both real misses and scorer blind spots.

## How to extend it

Add an item to `golden.json`: a `question`, the `expected_sources` (a substring of
the source note's label), and `expected_facts` (a string that must appear, or a
list where any one option counts). Re-run. That is the whole loop.
