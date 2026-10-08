"""Ask a model every question in the bank and write one results CSV.

Start here. Reads questions.csv, asks each question through tribunal/backend.py,
and writes results/<machine>_<model>_<style>_<timestamp>.csv with the run's
hardware and software details as leading '#' lines.

    python run.py --limit 3
    python run.py --model Qwen/Qwen3.5-9B --prompt-style permission
"""

import argparse
import csv
import hashlib
import os
import statistics
import subprocess
import sys
import zlib
from datetime import datetime
from pathlib import Path

# Never download here; weights come from download.py (compute nodes have no internet anyway).
os.environ["HF_HUB_OFFLINE"] = "1"
# transformers' threaded weight loading intermittently segfaults or hangs on MPS; load sequentially.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

from tribunal import DEFAULT_MODEL, REPO, hardware  # noqa: E402  (tribunal sets the cache location)
from tribunal.backend import Backend, pick_device, pick_dtype  # noqa: E402
import transformers  # noqa: E402
from huggingface_hub import constants, try_to_load_from_cache  # noqa: E402

PROMPTS = {
    "neutral": "Answer the question briefly.",
    "permission": "Answer the question briefly. If you are not sure, say so.",
}


def git_commit():
    """Short commit hash, suffixed '-dirty' if there are uncommitted changes; 'unknown' outside git."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"], capture_output=True, text=True)
    except FileNotFoundError:  # git isn't installed
        return "unknown"
    return out.stdout.strip() or "unknown"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--questions", type=Path, default=REPO / "questions.csv")
    p.add_argument("--limit", type=int, help="only ask the first N questions")
    p.add_argument("--prompt-style", choices=PROMPTS, default="neutral")
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], help="default: best available")
    p.add_argument("--device-map", choices=["single", "auto"], default="single",
                   help="'auto' splits a model too big for one GPU across all visible GPUs")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--samples", type=int, default=1, help="answers per question; >1 needs --temperature > 0")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default)")
    p.add_argument("--seed", type=int, default=0, help="makes sampled runs repeatable")
    p.add_argument("--out-dir", type=Path, default=REPO / "results")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.questions, newline="") as f:
        questions = list(csv.DictReader(f))[: args.limit]
    if not questions:
        sys.exit(f"No questions to ask: {args.questions} is empty or --limit is 0.")

    if args.samples > 1 and args.temperature == 0:
        sys.exit("--samples > 1 with greedy decoding would repeat the same answer; set --temperature too.")

    if not isinstance(try_to_load_from_cache(args.model, "config.json"), str):
        sys.exit(f"{args.model} is not in {constants.HF_HUB_CACHE}.\n"
                 f"Run first: python download.py --model {args.model}")

    device = args.device or pick_device()
    dtype = pick_dtype(device)
    print(f"Loading {args.model} on {device} ({dtype}); weights cache: {constants.HF_HUB_CACHE}")
    backend = Backend(args.model, device, dtype, args.device_map)
    print(f"Loaded in {backend.load_seconds:.1f}s\n")

    rows = []
    for q in questions:
        messages = [
            {"role": "system", "content": PROMPTS[args.prompt_style]},
            {"role": "user", "content": q["question"]},
        ]
        print(f"[{q['id']}] {q['question']}")
        # Reseed per question, from its id: sampled answers then don't shift when other questions
        # are added, reworded, or answered at a different length (e.g. under another prompt style).
        transformers.set_seed((args.seed + zlib.crc32(q["id"].encode())) % 2**32)
        for sample in range(1, args.samples + 1):
            response, new_tokens, seconds = backend.generate(messages, args.max_new_tokens, args.temperature)
            rows.append({
                **q,
                "prompt_style": args.prompt_style,
                "sample": sample,
                "response": response,
                "new_tokens": new_tokens,
                "seconds": round(seconds, 2),
                "tok_s": round(new_tokens / seconds, 1),
                "human_verdict": "",
            })
            print(f"  → {response}  ({new_tokens} tok, {new_tokens / seconds:.1f} tok/s)")
        print()

    info = hardware.describe(device, dtype)
    info |= {
        "transformers": transformers.__version__,
        "git_commit": git_commit(),
        # Fingerprint of the question bank, so runs can be compared only when they asked the same questions.
        "questions_sha": hashlib.sha256(args.questions.read_bytes()).hexdigest()[:12],
        "model": args.model,
        "prompt_style": args.prompt_style,
        "temperature": args.temperature,
        "samples": args.samples,
        "seed": args.seed,
        "load_seconds": round(backend.load_seconds, 1),
        # Median, not mean: the first question pays one-time setup and runs several times slower.
        "median_tok_s": statistics.median(r["tok_s"] for r in rows),
        "total_seconds": round(sum(r["seconds"] for r in rows), 1),
    }
    print("\n".join(f"{k:>14}: {v}" for k, v in info.items()))

    args.out_dir.mkdir(exist_ok=True)
    model_slug = args.model.split("/")[-1]
    out = args.out_dir / f"{hardware.machine_slug(info)}_{model_slug}_{args.prompt_style}_{datetime.now():%Y-%m-%dT%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        # Run metadata rides along as leading '#' lines. Skip them by count, not with pandas'
        # comment='#', which would also truncate any response containing '#'.
        f.writelines(f"# {k}: {v}\n" for k, v in info.items())
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out.resolve()}")


if __name__ == "__main__":
    main()
