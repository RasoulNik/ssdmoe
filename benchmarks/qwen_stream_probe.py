#!/usr/bin/env python3
"""Probe a local Qwen MoE checkpoint without loading routed experts.

This script is intentionally narrow: it validates that we can selectively load
only the non-expert text weights from the original safetensors shards. It is a
stepping stone toward a streamed routed-expert inference loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from streaming_qwen.model_io import (
    list_expert_tensors,
    list_non_expert_text_tensors,
    load_non_expert_text_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe selective loading for a local Qwen MoE snapshot"
    )
    parser.add_argument("--model", required=True, help="Local model snapshot path")
    parser.add_argument(
        "--load",
        action="store_true",
        help="Actually load the non-expert text weights into MLX",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()

    non_expert = list_non_expert_text_tensors(model_path)
    experts = list_expert_tensors(model_path)

    print(f"Model path: {model_path}")
    print(f"Non-expert text tensors: {len(non_expert)}")
    print(f"Expert tensors: {len(experts)}")

    if args.load:
        weights = load_non_expert_text_weights(model_path)
        total_bytes = sum(arr.nbytes for arr in weights.values())
        print(f"Loaded tensors: {len(weights)}")
        print(f"Loaded bytes: {total_bytes / 1024**3:.3f} GiB")


if __name__ == "__main__":
    main()
