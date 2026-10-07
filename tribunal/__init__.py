"""Importing this package points the Hugging Face cache at the repo's models/ folder,
unless HF_HOME or HF_HUB_CACHE is already set (e.g. to scratch space on a cluster).

Import it before transformers or huggingface_hub: they read the location once, at import.
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen3.5-2B"

if not (os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")):
    os.environ["HF_HUB_CACHE"] = str(REPO / "models")
