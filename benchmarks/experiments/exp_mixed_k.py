#!/usr/bin/env python3
"""Experiment with mixed K values across layers.

Test if we can use K=1 for some layers and K=2 for others
to balance quality and speed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.loader import ensure_src_path
ensure_src_path()

import mlx.core as mx
from mlx_lm.generate import stream_generate

from lib.index import load_index_config
from streaming_moe.runtime import build_streamed_model
from streaming_moe.streamed_switch import (
    STREAM_STATS,
    reset_stream_stats,
)


def set_layer_k(model, layer_indices: list[int], k: int):
    """Set top_k for specific layers."""
    for idx in layer_indices:
        if idx < len(model.layers):
            layer = model.layers[idx]
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "top_k"):
                layer.mlp.top_k = k


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
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--native-reader", default=".run/libexpert_reader.dylib")
    parser.add_argument("--output", default=".run/mixed-k-experiment.json")
    args = parser.parse_args()

    results = []

    # Configuration for mixed K experiments
    # Format: (name, k1_layers, k2_layers) where k1_layers use K=1, k2_layers use K=2
    # All other layers default to K=2
    num_layers = load_index_config(Path(args.index)).n_moe_layers
    half = num_layers // 2
    tail5 = min(5, half)
    experiments = [
        ("uniform-k2", [], list(range(num_layers))),  # All K=2 (baseline)
        ("uniform-k1", list(range(num_layers)), []),  # All K=1
        ("first-half-k1", list(range(half)), list(range(half, num_layers))),  # First half K=1, rest K=2
        ("last-half-k1", list(range(half, num_layers)), list(range(half))),  # Last half K=1, first half K=2
        ("alternate-k1k2", list(range(0, num_layers, 2)), list(range(1, num_layers, 2))),  # Alternating
        ("every-4th-k2", [i for i in range(num_layers) if i % 4 != 0], [i for i in range(num_layers) if i % 4 == 0]),
        ("first-last-5-k2", list(range(tail5, num_layers - tail5)),
         list(range(tail5)) + list(range(num_layers - tail5, num_layers))),
    ]

    for name, k1_layers, k2_layers in experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"  K=1 layers: {len(k1_layers)}, K=2 layers: {len(k2_layers)}")
        print(f"{'='*60}")

        mx.clear_cache()
        mx.reset_peak_memory()

        # Build model with default K=2
        model, tokenizer, expert_store, _ = build_streamed_model(
            model_path=Path(args.model),
            index_path=Path(args.index),
            top_k=2,  # Default
            native_reader_path=Path(args.native_reader) if args.native_reader else None,
        )

        try:
            # Set layer-specific K values
            set_layer_k(model, k1_layers, 1)
            set_layer_k(model, k2_layers, 2)

            # Warmup
            if args.warmup_tokens > 0:
                run_generation(model, tokenizer, args.prompt, args.warmup_tokens)
                mx.reset_peak_memory()

            # Benchmark
            expert_store.reset_stats()
            reset_stream_stats()
            final, text = run_generation(model, tokenizer, args.prompt, args.max_tokens)

            result = {
                "name": name,
                "k1_layers_count": len(k1_layers),
                "k2_layers_count": len(k2_layers),
                "generation_tps": final.generation_tps,
                "peak_memory_gb": final.peak_memory,
                "generation_tokens": final.generation_tokens,
                "text_preview": text[:300],
                "stream_stats": {
                    "selected_experts_total": STREAM_STATS["selected_experts_total"],
                    "calls": STREAM_STATS["calls"],
                },
            }
            if STREAM_STATS["calls"] > 0:
                result["avg_experts_per_call"] = STREAM_STATS["selected_experts_total"] / STREAM_STATS["calls"]

            results.append(result)
            print(f"  Generation: {final.generation_tps:.2f} tok/s")
            print(f"  Avg experts/call: {result.get('avg_experts_per_call', 0):.2f}")
            print(f"  Text: {text[:100]}...")

        finally:
            expert_store.close()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {output_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Name':<25} {'tok/s':>8} {'avg_k':>8}")
    print("-"*45)
    for r in results:
        print(f"{r['name']:<25} {r['generation_tps']:>8.2f} {r.get('avg_experts_per_call', 0):>8.2f}")


if __name__ == "__main__":
    main()
