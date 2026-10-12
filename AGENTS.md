# llm-tribunal: notes for coding agents

Teaching repo: readers are workshop attendees new to running LLMs.

## Writing code

- Comment density is higher than usual. Every module gets a short docstring saying what it does
  and where it sits; comment the conceptual hinges (chat templates, dtype, prompt-token slicing,
  device maps). Still never narrate a line the code already says.

## Running models

- `run.py` never downloads; fetch weights with `python download.py --model <id>` first.
- Smoke test with `--limit 3`. A full run (100 questions × 5 samples) on an M2 Pro takes about
  10 min for Qwen3.5-2B and 10–21 min for Qwen3.5-4B, depending on `--format`. Set timeouts from that.
- **Watching an unattended run:** detect failure from structure (the exit code, a timeout, status
  lines your own wrapper prints), never by searching `run.py`'s output. Model answers are free text
  and can contain "Error", "Traceback", or "Killed" ("ECC stands for Error-Correcting Code").
- Results CSVs stay local (`results/` is gitignored). Judge them with `python report.py results/*.csv`.
