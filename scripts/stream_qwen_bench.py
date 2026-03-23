#!/usr/bin/env python3
"""Benchmark streamed Qwen MoE generation across routed-expert top-k values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.utils import load as load_mlx_model

from streaming_qwen.runtime import build_streamed_model
from streaming_qwen.streamed_switch import (
    STREAM_STATS,
    reset_stream_stats,
    set_stream_profiling,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark streamed Qwen MoE")
    parser.add_argument("--model", required=True, help="Local model snapshot path")
    parser.add_argument("--index", required=True, help="Path to expert_index.json")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument(
        "--top-ks",
        default="8,6,4,3,2",
        help="Comma-separated routed expert counts to test",
    )
    parser.add_argument("--max-tokens", type=int, default=64, help="Generation length")
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=0,
        help="Optional warmup generation length to run before measurement",
    )
    parser.add_argument(
        "--warmup-prompt",
        default=None,
        help="Optional warmup prompt; defaults to the measured prompt",
    )
    parser.add_argument(
        "--output",
        default=".run/stream-qwen-bench.json",
        help="JSON output path",
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
    parser.add_argument(
        "--resident-small-components",
        action="store_true",
        help="Keep expert scales and biases resident while streaming expert weights",
    )
    parser.add_argument(
        "--component-workers",
        type=int,
        default=3,
        help="Concurrent native component-read workers",
    )
    parser.add_argument(
        "--profile-streamed",
        action="store_true",
        help="Synchronize inside streamed expert stages for timing breakdowns",
    )
    parser.add_argument(
        "--draft-model",
        default=None,
        help="Optional local draft model path for speculative decoding",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=2,
        help="Speculative decoding draft length when --draft-model is set",
    )
    return parser.parse_args()


def run_generation(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    draft_model=None,
    num_draft_tokens: int = 2,
):
    final = None
    text = ""
    for response in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        draft_model=draft_model,
        num_draft_tokens=num_draft_tokens,
    ):
        final = response
        text += response.text

    if final is None:
        raise RuntimeError("No generation response was produced")

    return final, text


def main() -> None:
    args = parse_args()
    top_ks = [int(x) for x in args.top_ks.split(",") if x.strip()]
    results = []
    draft_model = None
    if args.draft_model:
        draft_model, _ = load_mlx_model(args.draft_model, lazy=False)
        draft_model.eval()
        mx.eval(draft_model.parameters())

    for top_k in top_ks:
        mx.clear_cache()
        mx.reset_peak_memory()
        set_stream_profiling(args.profile_streamed)

        model, tokenizer, expert_store, _ = build_streamed_model(
            model_path=Path(args.model),
            index_path=Path(args.index),
            top_k=top_k,
            cache_limit_bytes=args.expert_cache_mb * 1024 * 1024,
            use_nocache=args.nocache,
            native_reader_path=Path(args.native_reader) if args.native_reader else None,
            resident_small_components=args.resident_small_components,
            component_workers=args.component_workers,
        )
        try:
            if args.warmup_tokens > 0:
                run_generation(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=args.warmup_prompt or args.prompt,
                    max_tokens=args.warmup_tokens,
                    draft_model=draft_model,
                    num_draft_tokens=args.num_draft_tokens,
                )
                mx.reset_peak_memory()

            expert_store.reset_stats()
            reset_stream_stats()
            final, text = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                draft_model=draft_model,
                num_draft_tokens=args.num_draft_tokens,
            )
        finally:
            expert_store.close()

        result = {
            "top_k": top_k,
            "prompt_tokens": final.prompt_tokens,
            "prompt_tps": final.prompt_tps,
            "generation_tokens": final.generation_tokens,
            "generation_tps": final.generation_tps,
            "peak_memory_gb": final.peak_memory,
            "finish_reason": final.finish_reason,
            "expert_store": expert_store.stats,
            "stream_stats": dict(STREAM_STATS),
            "text_preview": text[:500],
            "expert_cache_mb": args.expert_cache_mb,
            "nocache": args.nocache,
            "native_reader": args.native_reader,
            "profile_streamed": args.profile_streamed,
            "warmup_tokens": args.warmup_tokens,
            "warmup_prompt": args.warmup_prompt,
            "resident_small_components": args.resident_small_components,
            "component_workers": args.component_workers,
            "draft_model": args.draft_model,
            "num_draft_tokens": args.num_draft_tokens,
        }
        if expert_store.stats["read_seconds"] > 0:
            result["expert_read_gbps"] = (
                expert_store.stats["bytes_read"] / 1024**3
            ) / expert_store.stats["read_seconds"]
        else:
            result["expert_read_gbps"] = 0.0

        results.append(result)
        print(
            f"top_k={top_k} gen_tps={final.generation_tps:.3f} "
            f"peak_mem={final.peak_memory:.3f}GB "
            f"expert_gbps={result['expert_read_gbps']:.3f}"
        )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
