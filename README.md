# llm-tribunal

*A verdict for every answer a small LLM gives: does it **know**, **err**, **say "UNKNOWN"**, or **make things up**?*

A small, portable demo of running a small open-weight LLM (Qwen 3.5, 2B by default) on
whatever hardware you have (a laptop, a university cluster, an 8 GB gaming GPU), asking it
100 niche questions, and sorting the answers into verdicts. The goal is not
a leaderboard. The goal is to *see where a small model's knowledge gets lossy*, with code
short enough to read in one sitting.

> **Status: working demo.** Download a model, ask it the question bank, and get
> preliminary verdicts per category. An example SLURM script (`job.slurm`) is next.

## Quick start

```bash
git clone https://github.com/niznik-dev/llm-tribunal && cd llm-tribunal
conda env create -f environment.yml && conda activate llm-tribunal
python download.py            # Qwen3.5-2B, ~4 GB, into ./models
python run.py --limit 3       # ask 3 questions, write a CSV to results/
python report.py results/*.csv  # judge the answers and print verdict tables
```

Downloaded a zip instead? Everything works; results just record the commit as `unknown`.

Weights land in `./models/` unless `HF_HOME` or `HF_HUB_CACHE` is set. `run.py` never
downloads; for another model, run `python download.py --model <id>` first.

## Why "tribunal"?

In [Morrowind](https://en.uesp.net/wiki/Morrowind:Morrowind), the
[Tribunal](https://en.uesp.net/wiki/Lore:Tribunal) are three living gods who sit in
judgment. Here, the tribunal is a few lines of Python that pass a verdict on each
answer. If you have never played Morrowind, "tribunal" also just means a court, which
is close enough.

## The verdicts

Every question is either **real** or **fictional** (plausible-sounding bait that does not exist).

| Verdict | Question | What it looks like |
|---|---|---|
| **knows** | real | Gives the expected answer (or a listed alias). |
| **errs** | real | Answers with something else ("The pandas library was created by Monte Paninski"). |
| **says-unknown** | either | Abstains: the exact word offered (`exact`, "UNKNOWN") or its own phrasing (`synonym`, "I'm not sure"). |
| **rejects-premise** | fictional | Says the thing does not exist ("There is no NumPy function named `foldmerge`"). |
| **fabricates** | fictional | Answers as if the thing were real. |
| **refuses** | either | Declines to engage ("I can't help with that"). |

An answer cut off at `max_new_tokens` never counts as knowing: if it needed that many tokens,
it wasn't a short answer. Those are labeled `errs` or `fabricates` with the sub-label `filibuster`.

Verdicts come from simple, readable string heuristics in `tribunal/judge.py` and are explicitly
**preliminary**. `report.py` judges every answer fresh each time, so improving the rules
re-judges old runs too. `python report.py <file> --annotate` writes a copy with the verdicts
filled in next to an empty `human_verdict` column. Reviewing that column is part of the exercise.

## What the demo will show

1. **Where the lossiness lives.** Questions are grouped into categories a researcher might
   ask a chatbot about (coding, computer architecture, physical science, biological science)
   plus a category of plausible-sounding things that do not exist. The verdict mix per
   category is the finding.
2. **Whether permission changes behavior.** The same questions are run with and without a
   system prompt that offers the reply "UNKNOWN" when the model is not sure (`--prompt-style`).
3. **What your hardware does with it.** Every run records the chip or GPU, memory, device
   (`cuda` / `mps` / `cpu`), and tokens per second, so the same script produces a
   comparable row on any machine.

## Layout

```
llm-tribunal/
├── environment.yml      # conda: python, pytorch, transformers, accelerate
├── questions.csv        # the question bank (id, category, question, expected_answer, kind, notes)
├── download.py          # fetch a model's weights into ./models
├── run.py               # load model, ask every question, write a results CSV
├── report.py            # judge a run's answers, print verdict tables
├── tribunal/
│   ├── backend.py       # pick a device, load the model, generate
│   ├── judge.py         # the preliminary verdict heuristics
│   └── hardware.py      # what machine am I on?
├── models/              # downloaded weights (gitignored)
├── results/             # one CSV per run (machine, model, prompt, timestamp); stays local
└── job.slurm            # example batch script for a SLURM cluster (planned)
```

## Hardware tested

| Machine | Chip / GPU | Memory | Device | Model | tok/s |
|---|---|---|---|---|---|
| MacBook Pro 16" (2023) | Apple M2 Pro | 32 GB | `mps` | Qwen3.5-2B | *pending* |

## License

MIT. Model weights are downloaded from Hugging Face at run time under their own license
(Qwen 3.5: Apache-2.0) and are not part of this repository.
