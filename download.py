"""Fetch a model's weights into the cache (./models by default) so run.py can load them offline.

    python download.py
    python download.py --model Qwen/Qwen3.5-9B
"""

import argparse
from pathlib import Path

from tribunal import DEFAULT_MODEL  # sets the cache location; must precede huggingface_hub
from huggingface_hub import constants, snapshot_download


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    print(f"Downloading {args.model} → {constants.HF_HUB_CACHE}")
    path = Path(snapshot_download(args.model))
    gb = sum(f.stat().st_size for f in path.iterdir() if f.is_file()) / 2**30
    print(f"Done ({gb:.1f} GB). Now: python run.py --model {args.model}")


if __name__ == "__main__":
    main()
