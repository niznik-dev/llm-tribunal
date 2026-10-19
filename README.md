# llm-tribunal

*A verdict for every answer a small LLM gives: does it **know**, **err**, **say "UNKNOWN"**, or **make things up**?*

A small, portable demo of running a small open-weight LLM (Qwen 3.5, 2B by default) on
whatever hardware you have (a laptop, a university cluster, an 8 GB gaming GPU), asking it
100 niche questions, and sorting the answers into verdicts. The goal is not
a leaderboard. The goal is to *see where a small model's knowledge gets lossy*, with code
short enough to read in one sitting.

> **Status: working demo.** Download a model, ask it the question bank, and get
> preliminary verdicts per category. Example SLURM scripts run it on one GPU (`job.slurm`,
> tested on della) or four (`job-multigpu.slurm`, smoke-tested on della).

## Quick start

```bash
git clone https://github.com/niznik-dev/llm-tribunal && cd llm-tribunal
conda env create -f environment.yml && conda activate llm-tribunal
python download.py            # Qwen3.5-2B, ~4 GB, into ./models
python run.py --limit 3       # ask 3 questions, write a CSV to results/
python report.py results/*.csv  # judge the answers and print verdict tables
```

Downloaded a zip instead? Everything works; results just record the commit as `unknown`.

On an NVIDIA GPU, also run `pip install -r requirements-cuda.txt`. It adds optional fast
kernels for Qwen 3.5's linear-attention layers; without them, transformers warns that it is
using slower reference code. Skip it on a Mac or a machine without an NVIDIA GPU. The speedup
is modest, and one kernel (`causal_conv1d`) is left out for now; the file explains both.

Weights land in `./models/` unless `HF_HOME` or `HF_HUB_CACHE` is set. `run.py` never
downloads; for another model, run `python download.py --model <id>` first.

Already have weights as a plain folder (say, a shared copy on a cluster)? Pass the path:
`python run.py --model /path/to/Qwen3.5-27B`. A model is just a folder of config, tokenizer,
and weight files; the Hugging Face cache is bookkeeping on top. Each results file records the
model's repo id and revision, read from the cache or, for a folder, from an optional
`REVISION.txt` beside the weights containing `repo: <id>` and `revision: <commit>` lines.

## Running on a cluster

Two example SLURM scripts, written for Princeton's della and easy to adapt to other clusters:
`job.slurm` for one GPU, `job-multigpu.slurm` for four. Anything after the script name goes
straight to `run.py`:

```bash
sbatch job.slurm --model /path/to/Qwen3.8-27B --limit 3
sbatch job.slurm --model Qwen/Qwen3.5-9B --samples 5 --temperature 0.7
```

- **Download first, on a login node.** Compute nodes have no internet. Set `HF_HUB_CACHE`
  to a scratch directory (home quotas are small) before `python download.py --model <id>`,
  and keep the same setting in the job, either exported in your shell or uncommented in the script.
- **`#SBATCH` lines are defaults.** Flags passed to `sbatch` override them:
  `sbatch --time=2:00:00 job.slurm ...`.
- **More than one SLURM account?** della then requires you to pick one:
  `sbatch --account=<account> job.slurm ...`.
- **Email when the job starts or ends:** remove one `#` from the two `##SBATCH --mail-...`
  lines and fill in your NetID.
- **Did it work?** Check the exit code (`sacct -j <jobid>`) or the script's last line,
  `run.py exited with N`. Don't search the log for "Error": the log holds the model's answers,
  and those can contain any word.

**Sizing a model to GPUs.** In bf16 every parameter takes 2 bytes, so a 27B model needs
~54 GB (fits on one 80 GB A100), and a 122B model needs ~233 GiB, which takes four (320 GiB
total, leaving room for activations and the KV cache). `job-multigpu.slurm` passes
`--device-map auto`, which puts consecutive layers on consecutive GPUs. The GPUs take
turns rather than working in parallel, so extra GPUs buy room for the model, not speed.
Multi-GPU jobs also tend to wait longer in the queue: the scheduler has to find a whole
free node.

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
├── requirements-cuda.txt # optional: fast kernels, NVIDIA GPUs only
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
├── job.slurm            # example SLURM batch script, one GPU
└── job-multigpu.slurm   # the same for a model split across four GPUs
```

## Hardware tested

| Machine | Chip / GPU | Memory | Device | Model | tok/s |
|---|---|---|---|---|---|
| MacBook Pro 16" (2023) | Apple M2 Pro | 32 GB | `mps` | Qwen3.5-2B | *pending* |
| Princeton della | NVIDIA A100 80GB PCIe | 80 GB | `cuda` | Qwen3.8-27B | 14 |
| Princeton della | 4× NVIDIA A100-SXM4-80GB | 4 × 80 GB | `cuda` | Qwen3.5-122B-A10B | 10 (3-question smoke test) |

## License

MIT. Model weights are downloaded from Hugging Face at run time under their own license
(Qwen 3.5: Apache-2.0) and are not part of this repository.
