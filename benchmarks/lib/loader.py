"""Shared utilities for benchmark scripts.

Usage in any benchmark:
    from lib.loader import ensure_src_path, parse_bytes, save_json
    ensure_src_path()  # call before importing streaming_qwen
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def ensure_src_path() -> None:
    """Add repo src/ to sys.path so streaming_qwen is importable without install."""
    src = str(Path(__file__).resolve().parent.parent.parent / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def parse_bytes(s: str) -> int:
    """Parse a human-readable byte string like '1G', '500M', '2048' into bytes."""
    suffixes = {
        "GB": 1_073_741_824, "G": 1_073_741_824,
        "MB": 1_048_576,     "M": 1_048_576,
        "KB": 1024,          "K": 1024,
        "B": 1,              "": 1,
    }
    s = s.strip()
    for suffix in sorted(suffixes, key=len, reverse=True):
        if suffix and s.upper().endswith(suffix):
            return int(float(s[: -len(suffix)]) * suffixes[suffix])
    return int(float(s))


def save_json(data: object, path: str | Path | None) -> None:
    """Write data as indented JSON; no-op if path is None."""
    if path is None:
        return
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out}")
