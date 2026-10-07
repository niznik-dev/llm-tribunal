"""Ask a model every question in the bank and write one results CSV.

    python run.py --limit 3
    python run.py --model Qwen/Qwen3.5-9B --prompt-style permission
"""

import argparse
import csv
import os
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent

# Must happen before transformers is imported: it reads the cache location at import time.
if not (os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")):
    os.environ["HF_HUB_CACHE"] = str(REPO / "models")

from tribunal import hardware  # noqa: E402
from tribunal.backend import Backend, pick_device, pick_dtype  # noqa: E402

PROMPTS = {
    "neutral": "Answer the question briefly.",
    "permission": "Answer the question briefly. If you are not sure, say so.",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--questions", type=Path, default=REPO / "questions.csv")
    p.add_argument("--limit", type=int, help="only ask the first N questions")
    p.add_argument("--prompt-style", choices=PROMPTS, default="neutral")
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], help="default: best available")
    p.add_argument("--device-map", choices=["single", "auto"], default="single",
                   help="'auto' splits a model too big for one GPU across all visible GPUs")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--out-dir", type=Path, default=REPO / "results")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.questions, newline="") as f:
        questions = list(csv.DictReader(f))[: args.limit]

    device = args.device or pick_device()
    dtype = pick_dtype(device)
    print(f"Loading {args.model} on {device} ({dtype}); weights cache: {os.environ.get('HF_HUB_CACHE') or os.environ['HF_HOME']}")
    backend = Backend(args.model, device, dtype, args.device_map)
    print(f"Loaded in {backend.load_seconds:.1f}s\n")

    rows = []
    for q in questions:
        messages = [
            {"role": "system", "content": PROMPTS[args.prompt_style]},
            {"role": "user", "content": q["question"]},
        ]
        response, new_tokens, seconds = backend.generate(messages, args.max_new_tokens)
        rows.append({
            **q,
            "prompt_style": args.prompt_style,
            "response": response,
            "new_tokens": new_tokens,
            "seconds": round(seconds, 2),
            "tok_s": round(new_tokens / seconds, 1),
            "human_verdict": "",
        })
        print(f"[{q['id']}] {q['question']}\n  → {response}\n  ({new_tokens} tok, {new_tokens / seconds:.1f} tok/s)\n")

    info = hardware.describe(device, dtype)
    info |= {
        "model": args.model,
        "prompt_style": args.prompt_style,
        "load_seconds": round(backend.load_seconds, 1),
        "mean_tok_s": round(sum(r["new_tokens"] for r in rows) / sum(r["seconds"] for r in rows), 1),
        "total_seconds": round(sum(r["seconds"] for r in rows), 1),
    }
    print("\n".join(f"{k:>14}: {v}" for k, v in info.items()))

    args.out_dir.mkdir(exist_ok=True)
    model_slug = args.model.split("/")[-1]
    out = args.out_dir / f"{hardware.machine_slug(info)}_{model_slug}_{args.prompt_style}_{date.today()}.csv"
    with open(out, "w", newline="") as f:
        # Run metadata rides along as leading '#' lines. Skip them by count, not with pandas'
        # comment='#', which would also truncate any response containing '#'.
        f.writelines(f"# {k}: {v}\n" for k, v in info.items())
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    sys.exit(main())
