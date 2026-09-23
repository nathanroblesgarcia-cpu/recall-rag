# What my evals caught: building a small agent that checks its notes and its numbers

In my first write-up I built Recall, a small app that answers questions from my own
notes. This one is about the next step: turning it into an **agent** that can also
look up live numbers, and what happened when I tested it properly.

Short version: the agent looked fine when I tried it by hand. Then I gave it a
quiz, and the quiz caught it making things up. Four different ways. Most of the fixes
ended up in code, not in a better prompt.

The code is on GitHub
([recall-rag](https://github.com/nathanroblesgarcia-cpu/recall-rag)). It ships
with a made-up café so anyone can run it, and the examples below are described
in general terms or taken from that café.

## Step 1: better search first

Before the agent, I fixed the search underneath it.

Recall found notes by *meaning*. You turn text into numbers that stand for its
meaning, then find the notes with the closest numbers. It's great when words
differ ("leave" finds "holiday"), but it can blur exact things: a supplier name,
a date, a short code.

So I added a second search next to it: plain **exact-word search**, which SQLite
has built in (FTS5). Each question now runs both searches, and the two ranked lists
are merged. A note that ranks high in both goes to the top. The merge method is
called Reciprocal Rank Fusion, and it only looks at *positions*, so the two
searches' very different scores never need comparing.

Two lessons came out of this.

**Pass/fail hides rank.** On a larger note set, the old search already "found"
nearly every test question. It looked done. But the answer model only reads the
top few notes, and some right notes sat near the bottom of that list. So I started
tracking *where* the right note lands (MRR, where 1.0 means always first). Hybrid
search moved it up, and the one miss became a hit.

**A tiny demo can't prove an upgrade.** On the 16-note café demo, both searches
score perfectly. Each question has one obvious note, so there is nothing to
rescue. A demo proves the wiring works. Only messy real data proves an upgrade
helps.

I also tried **reranking**: a second model rereads the shortlist and sorts it
again. It nudged the rank score up a little, didn't change which notes reached the
answer model, and made each search much slower. So it's in the code,
switched off, with the numbers written next to the setting. A measured "no" is
still a result.

## Step 2: one agent, two kinds of tool

The agent is a loop. The model reads the question and decides what it needs:

- **search_notes** for anything written down (goals, plans, suppliers, recipes)
- **number tools** for live figures (sales, costs, profit, balances)
- or **both**, for a question like *"How much is in the emergency fund now, and
  what is my target?"*

A model can only write text. It can't actually *do* anything. So when it wants a
tool, it asks, my code runs the tool, and the result goes back in front of it.
Then it writes the answer.

On the web page you can watch it work:

1. Looked up the café's cash and fund balances
2. Searched your notes for "emergency fund target"

> The current balance in the emergency fund is $14,400. Your target for the
> emergency fund is $25,000.

When I tried it by hand, it felt finished.

## Step 3: the quiz

I wrote a small quiz for the agent: questions that need only notes, only numbers,
or both. Each answer is checked three ways:

1. **Routing:** did it call the tools the question needs?
2. **Facts:** does the answer contain the known facts from the notes?
3. **Grounded numbers:** does the answer quote a money figure that matches a
   number a tool *actually returned in that run*? Live numbers change, so the
   check compares against the tool's own output, never a number I typed in.

The first run scored 7 out of 9. Here is what the failures were, because they are
the whole point.

## What it caught

**1. It answered without looking, then invented a note.** Asked why something
was blank, the agent didn't search at all. It wrote a confident answer and cited a
note with a title and a date. That note does not exist. This is the famous AI
failure, "confidently wrong", and it happened on my own small project, in a quiz of
nine questions.

*Fix:* if the first answer uses no tool at all, the code sends it back once:
"You have not looked anything up yet, so you do not know the answer."

**2. It wrote the tool call instead of making it.** On a rerun, the model *tried*
to search, but it typed the search out as text, like
`search_notes({"query": "..."})`, instead of making a real tool call. My loop saw
text and treated it as the final answer. Small local models do this sometimes, in
several different formats.

*Fix:* if the whole reply is just a tool call written as text, and it names one of
my own tools, the code runs it for real. My first version of this missed a format,
so the quiz failed it three times in a row. Three failures in a row pointed at a
pattern, not bad luck.

**3. It changed a date.** A note gave a single year, say 2040. The agent turned it
into a whole decade, "the 2040s". And my quiz *passed* it, because "2040" is
inside "2040s". So the quiz was wrong too.

*Fixes:* a rule to copy numbers and dates exactly, and a "forbidden answers" list
in the quiz, so a near-miss like "2040s" now fails. One honest detail: my first
version of the rule used this exact date as its example. That's teaching to the
test. With a neutral example it passes 2 out of 3 tries, so I record that as a
known limit of a small model, not a solved problem.

**4. It claimed a search it never ran.** On the café version, the agent said
*"I searched the notes but could not find any information"*. It hadn't searched.
Another time it looked up the balance, then said the target was "not mentioned in
the notes", again without searching.

*Fixes, in code:* if it still won't use a tool after being sent back, the code
runs the note search for it and puts the results in front of it. And if an answer
talks about "the notes" but no search happened, it gets sent back to search.

## What I took from it

**Hand-testing lies to you.** Every failure above happened on a question I would
have happily tried by hand. I only saw them because the quiz ran every question,
every time, and checked the answer against the tools' real output.

**Fix it in code when you can.** A prompt is a polite request. The model ignored
"always search first" more than once. A check in the code that won't accept an
answer until a search has happened doesn't depend on the model agreeing.

**Check the checker.** The quiz had its own bug (the decade pass). If you never
read the answers yourself, you're trusting a scorer you haven't tested.

**Small models are fine for this, with guards.** Everything here runs on a laptop
with no graphics card, on a 7-billion-parameter model. It's slow, about one to two
minutes for a question that needs both notes and numbers. With the guards it
answers the café quiz correctly, 9 of 9. Without them, it didn't.

If you build one of these, write the quiz before you believe the demo.
