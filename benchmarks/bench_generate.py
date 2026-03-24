#!/usr/bin/env python3
"""Benchmark MoE generation throughput across routed-expert top-k values.

Works with any model supported by build_streamed_model (Qwen3 MoE, Nemotron-H, ...).
Sweeps --top-ks and reports tok/s, peak memory, and expert read bandwidth per run.

Examples:
  # Qwen3.5-35B-A3B on port 9002 index
  poetry run python benchmarks/bench_generate.py \\
    --model ~/.cache/huggingface/hub/.../Qwen3.5-35B-A3B-4bit/snapshots/<hash> \\
    --index .run/qwen35b-expert-index.json \\
    --native-reader .run/libexpert_reader.dylib \\
    --prompt "Explain MoE routing." --top-ks 8,6,4,2

  # Nemotron-H 30B
  poetry run python benchmarks/bench_generate.py \\
    --model ~/.cache/huggingface/hub/.../NVIDIA-Nemotron-3-Nano-30B-A3B-4bit/snapshots/<hash> \\
    --index .run/nemotron30b-expert-index.json \\
    --native-reader .run/libexpert_reader.dylib \\
    --prompt "Explain SSM vs transformer." --top-ks 6,4,2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lib.loader import save_json  # noqa: E402

import mlx.core as mx  # noqa: E402
from mlx_lm.generate import stream_generate  # noqa: E402
from mlx_lm.utils import load as load_mlx_model  # noqa: E402

from streaming_moe.runtime import build_streamed_model  # noqa: E402
from streaming_moe.streamed_switch import (  # noqa: E402
    STREAM_STATS,
    reset_stream_stats,
    set_stream_profiling,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark MoE generation throughput")
    p.add_argument("--model", required=True, help="Local model snapshot path")
    p.add_argument("--index", required=True, help="Path to expert_index.json")
    p.add_argument("--prompt", required=True, help="Prompt text")
    p.add_argument("--top-ks", default="8,6,4,3,2", help="Comma-separated top-k values to sweep")
    p.add_argument("--max-tokens", type=int, default=64, help="Generation length per run")
    p.add_argument("--warmup-tokens", type=int, default=0, help="Warmup generation length (excluded from metrics)")
    p.add_argument("--warmup-prompt", default=None, help="Warmup prompt (defaults to --prompt)")
    p.add_argument("--native-reader", default=None, help="Path to native expert reader dylib")
    p.add_argument("--component-workers", type=int, default=3, help="Concurrent native component-read workers")
    p.add_argument("--moe-impl", choices=["streamed", "pipelined"], default="streamed",
                   help="MoE implementation: streamed (default) or pipelined")
    p.add_argument("--fused-gate-up", action="store_true", help="Use fused gate/up/swiglu expert path")
    p.add_argument("--compile-fused-gate-up", action="store_true", help="mx.compile for fused gate/up path")
    p.add_argument("--resident-small-components", action="store_true",
                   help="Keep expert scales/biases resident; stream only weights")
    p.add_argument("--expert-cache-mb", type=int, default=0, help="In-process LRU expert cache in MiB")
    p.add_argument("--nocache", action="store_true", help="Open expert shards with F_NOCACHE")
    p.add_argument("--draft-model", default=None, help="Draft model path for speculative decoding")
    p.add_argument("--num-draft-tokens", type=int, default=2, help="Speculative draft length")
    p.add_argument("--profile", action="store_true", help="Force mx.eval after each MoE sub-op (lowers tok/s)")
    p.add_argument("--output", default=".run/bench-generate.json", help="JSON output path")
    return p.parse_args()


def _generate(model, tokenizer, prompt, max_tokens, draft_model=None, num_draft_tokens=2):
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
        raise RuntimeError("No generation response produced")
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
        set_stream_profiling(args.profile)

        model, tokenizer, expert_store, _ = build_streamed_model(
            model_path=Path(args.model),
            index_path=Path(args.index),
            top_k=top_k,
            cache_limit_bytes=args.expert_cache_mb * 1024 * 1024,
            use_nocache=args.nocache,
            native_reader_path=Path(args.native_reader) if args.native_reader else None,
            resident_small_components=args.resident_small_components,
            component_workers=args.component_workers,
            moe_impl=args.moe_impl,
            fused_gate_up=args.fused_gate_up,
            compile_fused_gate_up=args.compile_fused_gate_up,
        )
        try:
            if args.warmup_tokens > 0:
                _generate(model, tokenizer, args.warmup_prompt or args.prompt,
                          args.warmup_tokens, draft_model, args.num_draft_tokens)
                mx.reset_peak_memory()

            expert_store.reset_stats()
            reset_stream_stats()
            final, text = _generate(model, tokenizer, args.prompt, args.max_tokens,
                                     draft_model, args.num_draft_tokens)
        finally:
            expert_store.close()

        read_s = expert_store.stats["read_seconds"]
        expert_gbps = (
            (expert_store.stats["bytes_read"] / 1024**3) / read_s if read_s > 0 else 0.0
        )
        result = {
            "top_k": top_k,
            "moe_impl": args.moe_impl,
            "prompt_tokens": final.prompt_tokens,
            "prompt_tps": final.prompt_tps,
            "generation_tokens": final.generation_tokens,
            "generation_tps": final.generation_tps,
            "peak_memory_gb": final.peak_memory,
            "finish_reason": final.finish_reason,
            "expert_read_gbps": expert_gbps,
            "expert_store": expert_store.stats,
            "stream_stats": dict(STREAM_STATS),
            "text_preview": text[:500],
        }
        results.append(result)
        print(
            f"top_k={top_k} impl={args.moe_impl} "
            f"gen_tps={final.generation_tps:.2f} "
            f"peak_mem={final.peak_memory:.2f}GB "
            f"expert_gbps={expert_gbps:.2f}"
        )

    save_json(results, args.output)


if __name__ == "__main__":
    main()
