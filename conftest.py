"""Pytest bootstrap: put the repo root on sys.path so `import src.*`, `dpo.*`,
`sft.*`, `distill.*` resolve when tests run from anywhere. Mirrors the sys.path
insertion each training script does for itself."""

from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
