#!/usr/bin/env python3
"""Benchmark pipelined MoE against baseline streamed MoE."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import stream_generate

from streaming_qwen.runtime import build_streamed_model
from streaming_qwen.pipelined_moe import patch_pipelined_moe, reset_pipelined_stats
from streaming_qwen.streamed_switch import (
    STREAM_STATS,
    reset_stream_stats,
    set_stream_profiling,
)


def run_generation(model, tokenizer, prompt: str, max_tokens: int):
    final = None
    text = ""
    for response in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    ):
        final = response
        text += response.text
    if final is None:
        raise RuntimeError("No generation response")
    return final, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--prompt", default="Explain what a mixture-of-experts model is in two concise sentences.")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--native-reader", default=".run/libexpert_reader.dylib")
    parser.add_argument("--component-workers", type=int, default=3)
    parser.add_argument("--top-ks", default="2,4,8")
    parser.add_argument("--output", default=".run/pipelined-bench.json")
    parser.add_argument("--pipelined", action="store_true", help="Use pipelined MoE")
    args = parser.parse_args()

    top_ks = [int(x) for x in args.top_ks.split(",") if x.strip()]
    results = []

    for top_k in top_ks:
        print(f"\n{'='*60}")
        print(f"Testing top_k={top_k}, pipelined={args.pipelined}")
        print(f"{'='*60}")

        mx.clear_cache()
        mx.reset_peak_memory()

        model, tokenizer, expert_store, config = build_streamed_model(
            model_path=Path(args.model),
            index_path=Path(args.index),
            top_k=top_k,
            native_reader_path=Path(args.native_reader) if args.native_reader else None,
            component_workers=args.component_workers,
        )

        try:
            if args.pipelined:
                print("Applying pipelined MoE patch...")
                quantization = config.get("quantization") or config.get("quantization_config") or {}
                patch_pipelined_moe(model, expert_store, quantization)
                reset_pipelined_stats()

            # Warmup
            if args.warmup_tokens > 0:
                print(f"Warmup with {args.warmup_tokens} tokens...")
                run_generation(model, tokenizer, args.prompt, args.warmup_tokens)
                mx.reset_peak_memory()

            # Benchmark
            expert_store.reset_stats()
            reset_stream_stats()
            print("Running benchmark...")
            t0 = time.perf_counter()
            final, text = run_generation(model, tokenizer, args.prompt, args.max_tokens)
            total_time = time.perf_counter() - t0

            result = {
                "top_k": top_k,
                "pipelined": args.pipelined,
                "prompt_tokens": final.prompt_tokens,
                "prompt_tps": final.prompt_tps,
                "generation_tokens": final.generation_tokens,
                "generation_tps": final.generation_tps,
                "peak_memory_gb": final.peak_memory,
                "total_time_s": total_time,
                "expert_store": expert_store.stats,
                "stream_stats": dict(STREAM_STATS),
                "text_preview": text[:200],
            }
            if expert_store.stats["read_seconds"] > 0:
                result["expert_read_gbps"] = (
                    expert_store.stats["bytes_read"] / 1024**3
                ) / expert_store.stats["read_seconds"]
            else:
                result["expert_read_gbps"] = 0.0

            results.append(result)
            print(f"  Generation: {final.generation_tps:.2f} tok/s")
            print(f"  Peak memory: {final.peak_memory:.2f} GB")
            print(f"  Expert read: {result['expert_read_gbps']:.2f} GB/s")
            print(f"  Text: {text[:100]}...")

        finally:
            expert_store.close()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
