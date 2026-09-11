# llm-tribunal

*Four verdicts for every answer a small LLM gives: **knows**, **refuses**, **says "I don't know"**, or **hallucinates**.*

A small, portable demo of running a small open-weight LLM (Qwen 3.5, 2B by default) on
whatever hardware you have (a laptop, a university cluster, an 8 GB gaming GPU), asking it
~100 niche questions, and sorting the answers into the four verdicts above. The goal is not
a leaderboard. The goal is to *see where a small model's knowledge gets lossy*, with code
short enough to read in one sitting.

> **Status: skeleton.** The plan is settled; the code is coming next. Watch this space.

## Why "tribunal"?

In Morrowind, the Tribunal are three living gods who sit in judgment. Here, the tribunal
is a few lines of Python that pass one of four verdicts on each answer. If you have never
played Morrowind, "tribunal" also just means a court, which is close enough.

## The four verdicts

| Verdict | What it looks like |
|---|---|
| **knows** | Gives the expected answer (or a close alias). |
| **refuses** | Declines to answer ("I can't help with that"). |
| **says IDK** | Admits uncertainty ("I'm not sure", "I have no information on that"). |
| **hallucinates** | Answers confidently and wrongly, including about things that do not exist. |

Verdicts are assigned by simple, readable string heuristics and are explicitly
**preliminary**. Every results file carries an empty column for a human to overrule them.
Reviewing that column is part of the exercise.

## What the demo will show

1. **Where the lossiness lives.** Questions are grouped into categories that people *expect*
   a chatbot to know (long-tail geography, deep-cut media characters, mid-tier dates,
   women's sports records, ...) plus a category of plausible-sounding things that do not
   exist. The verdict mix per category is the finding.
2. **Whether permission changes behavior.** The same questions are run with and without a
   system prompt that says "if you are not sure, say so."
3. **What your hardware does with it.** Every run records the chip or GPU, memory, device
   (`cuda` / `mps` / `cpu`), and tokens per second, so the same script produces a
   comparable row on any machine.

## Planned layout

```
llm-tribunal/
├── environment.yml      # conda: python, pytorch, transformers, accelerate
├── questions.csv        # the question bank (id, category, question, expected_answer, kind, notes)
├── run.py               # load model, ask every question, write a results CSV
├── tribunal/
│   ├── backend.py       # pick a device, load the model, generate
│   ├── judge.py         # the preliminary verdict heuristics
│   ├── hardware.py      # what machine am I on?
│   └── report.py        # summary table
├── results/             # one CSV per (machine, model, date), plus a cross-hardware table
└── job.slurm            # example batch script for a SLURM cluster
```

## Hardware tested

| Machine | Chip / GPU | Memory | Device | Model | tok/s |
|---|---|---|---|---|---|
| MacBook Pro 16" (2023) | Apple M2 Pro | 32 GB | `mps` | Qwen3.5-2B | *pending* |

## License

MIT. Model weights are downloaded from Hugging Face at run time under their own license
(Qwen 3.5: Apache-2.0) and are not part of this repository.
