#!/usr/bin/env python3
"""Run streamed Qwen MoE generation with adjustable routed expert top-k."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlx_lm.generate import generate

from streaming_qwen.runtime import build_streamed_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate with streamed Qwen MoE")
    parser.add_argument("--model", required=True, help="Local model snapshot path")
    parser.add_argument("--index", required=True, help="Path to expert_index.json")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max generation tokens")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override routed experts per token",
    )
    parser.add_argument(
        "--expert-cache-mb",
        type=int,
        default=0,
        help="Bounded in-process expert cache size in MiB",
    )
    parser.add_argument(
        "--nocache",
        action="store_true",
        help="Open expert shard fds with F_NOCACHE",
    )
    parser.add_argument(
        "--native-reader",
        default=None,
        help="Path to native expert reader dylib",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, expert_store, _ = build_streamed_model(
        model_path=Path(args.model),
        index_path=Path(args.index),
        top_k=args.top_k,
        cache_limit_bytes=args.expert_cache_mb * 1024 * 1024,
        use_nocache=args.nocache,
        native_reader_path=Path(args.native_reader) if args.native_reader else None,
    )
    try:
        generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            verbose=True,
        )
    finally:
        expert_store.close()


if __name__ == "__main__":
    main()
