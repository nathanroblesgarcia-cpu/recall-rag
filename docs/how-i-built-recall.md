# How I built Recall: a local RAG app with an eval suite

I build data models and dashboards for a living, and I kept losing track of my own
notes: decisions, config, client quirks, all scattered across folders. So I built
Recall, a small app that lets me ask a plain question and get an answer grounded in
my own notes, with the source named. It runs entirely on my laptop.

This is the write-up of how it works and what I learned. The code is on GitHub, and
it ships with a fictional sample note set so you can run it without any of my data.

## What "RAG" actually means

RAG stands for retrieval-augmented generation. Stripped of the jargon, it is one
idea: **do not ask the model to answer from memory. First find the relevant notes,
then give them to the model and say "answer using these."**

That one change fixes the two biggest problems with asking a model directly. The
model stops making things up, because it is answering from real text in front of
it, and it can cite where the answer came from, because it knows which notes it was
handed.

## The three parts

**1. Ingest.** Read every note, cut each one into chunks of about 800 characters,
and turn each chunk into an embedding: a list of 384 numbers that captures its
meaning. Store the chunks and their numbers in a single SQLite file. I use
fastembed with the `bge-small` model, which runs on the machine, so no text is
sent anywhere to build the index.

**2. Retrieve.** When a question comes in, embed it the same way, then compare it
against every stored chunk with a cosine similarity and take the closest few.
"Closest" here means closest in meaning, not matching keywords, so "how much do I
save each month" finds the savings note even if it never uses the word "save."

**3. Generate.** Hand those chunks to an answer model with the question and an
instruction to answer only from the provided notes. By default this is a small
local model (`llama3.2:3b`) running under Ollama, so the whole loop stays on the
machine and free. Swapping to Claude in the cloud is a one-line config change.

## The part that mattered most: evals

The first version felt great, because I only ever tried it on questions I knew it
could answer. That is the trap. The moment you change the chunk size, or the number
of chunks you retrieve, or the model, you need to know whether you made it better
or worse. You cannot feel that. You have to measure it.

So I wrote a small eval suite. It is a fixed set of questions whose answers I know
(the "golden set"), and a scorer that checks two things separately: did the right
note get retrieved, and did the answer actually contain the known facts. It writes
a timestamped report every run, so I can compare before and after any change.

Then I added an LLM-as-judge: a second, stronger local model that reads the
question and the answer and gives a verdict. This catches answers that are correct
but phrased differently, which a plain text match would wrongly fail. Comparing the
two scorers against each other is how I found both real misses and blind spots in
my own scoring.

## What I would do next

- Chunk on markdown headings instead of blank lines, so a chunk is always a whole
  section.
- Rewrite the question before retrieving, to expand short questions.
- Add a re-ranking pass over the top chunks before answering. (Done since, and
  measured: it was not worth its cost here. See
  [What my evals caught](what-my-evals-caught.md), which also covers hybrid search
  and the agent.)
- Move the search into Postgres with pgvector once the note count outgrows loading
  every embedding into memory.

## What building it taught me

The retrieval and the model are the easy, fun part. The eval suite is the part that
turns it from a toy into something I actually trust, and it is the part most demos
skip. If I had to keep one file from this project, it would be the golden set.
