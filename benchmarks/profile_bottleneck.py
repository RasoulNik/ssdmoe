#!/usr/bin/env python3
"""Bottleneck profiler — break decode time into its component buckets.

Reports wall-clock time per token split into:
  read      — pread() inside native reader (expert_store.stats)
  convert   — bytes → numpy → mx.array (STREAM_STATS["convert_seconds"])
  remap     — expert index remapping (STREAM_STATS["remap_seconds"])
  compute   — gather_qmm × 3 + swiglu (measured with forced mx.eval per op)
  other     — attention + shared_expert + non-MoE layers + Python + sampling

Two passes are run:
  pass 1: realistic (lazy eval, no forced sync inside MoE) — gives real tok/s
  pass 2: profiled (forced mx.eval after each MoE sub-op) — gives compute breakdown
          NOTE: pass 2 tok/s is slower than reality due to forced sync overhead

Usage:
  poetry run python benchmarks/profile_bottleneck.py \\
      --model <model_path> --index <index_path> --native-reader <dylib_path> \\
      [--k 4] [--max-tokens 64] [--cache] [--window-tokens 2]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming_qwen.runtime import (
    build_streamed_model,
    begin_session_cache_request,
    complete_session_cache_token,
    end_session_cache_request,
    set_session_cache_phase,
    collect_session_cache_stats,
    set_routed_top_k,
)
from streaming_qwen.session_window_cache import SessionWindowNativeCache
from streaming_qwen.streamed_switch import (
    STREAM_STATS,
    set_stream_profiling,
    StreamedSwitchGLU,
)
from streaming_qwen.server.protocol import prompt_tokens_from_messages

PROMPT = "Output the word hello exactly 160 times separated by spaces, and nothing else."
SKIP = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--native-reader", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--cache", action="store_true", help="Enable session-window cache H=window-tokens")
    p.add_argument("--window-tokens", type=int, default=2)
    p.add_argument("--fused", action="store_true", help="Use fused gate+up path")
    return p.parse_args()


def patch_switches(model, expert_store, *, fused: bool, session_cache, quantization: dict, top_k: int):
    from streaming_qwen.runtime import iter_moe_layers
    for layer_idx, layer in enumerate(iter_moe_layers(model)):
        mlp = layer.mlp
        if not hasattr(mlp, "switch_mlp"):
            continue
        mlp.switch_mlp = StreamedSwitchGLU(
            layer_idx=layer_idx,
            expert_store=expert_store,
            group_size=quantization.get("group_size", 64),
            bits=quantization.get("bits", 4),
            mode=quantization.get("mode", "affine"),
            fused_gate_up=fused,
            compile_fused_gate_up=False,
            session_cache=session_cache,
        )
    set_routed_top_k(model, top_k)


def run_pass(model, tokenizer, expert_store, config, prompt_tokens, *,
             session_cache, top_k, max_tokens, profiled: bool, label: str):
    expert_store.reset_stats()
    for k in list(STREAM_STATS.keys()):
        STREAM_STATS[k] = 0

    set_stream_profiling(profiled)

    prompt_cache = make_prompt_cache(model)
    if session_cache is not None:
        begin_session_cache_request(model, session_id="prof", phase="prefill",
                                    enabled=True, ephemeral=True)
    prompt = mx.array(prompt_tokens, dtype=mx.uint32)
    if prompt.size > 1:
        model(prompt[:-1][None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])
        mx.clear_cache()
    last = prompt[-1:]

    expert_store.reset_stats()
    for k in list(STREAM_STATS.keys()):
        STREAM_STATS[k] = 0
    mx.reset_peak_memory()
    if session_cache is not None:
        set_session_cache_phase(model, "decode")

    token_times = []
    started = time.perf_counter()
    for token, _ in generate_step(last, model, max_tokens=max_tokens, prompt_cache=prompt_cache):
        token_times.append(time.perf_counter())
        if session_cache is not None:
            complete_session_cache_token(model)

    if session_cache is not None:
        end_session_cache_request(model)

    set_stream_profiling(False)

    total_s = token_times[-1] - started if token_times else 0
    n = len(token_times)
    n_measured = n - SKIP
    wall_s = token_times[-1] - token_times[SKIP - 1] if n > SKIP else total_s

    read_s = expert_store.stats["read_seconds"]
    convert_s = STREAM_STATS["convert_seconds"]
    remap_s = STREAM_STATS["remap_seconds"]
    qmm_s = (STREAM_STATS["qmm_up_seconds"] + STREAM_STATS["qmm_gate_seconds"]
              + STREAM_STATS["qmm_down_seconds"] + STREAM_STATS["swiglu_seconds"])
    other_s = max(0.0, wall_s - read_s - convert_s - remap_s - (qmm_s if profiled else 0))

    ssd_gb = expert_store.stats["bytes_read"] / 1e9
    tps = n_measured / wall_s if wall_s > 0 else 0

    sc = collect_session_cache_stats(model) if session_cache else None

    return {
        "label": label,
        "profiled": profiled,
        "tok_s": round(tps, 3),
        "n_tokens": n,
        "wall_s": round(wall_s, 3),
        "ssd_gb": round(ssd_gb, 2),
        "read_s": round(read_s, 3),
        "convert_s": round(convert_s, 3),
        "remap_s": round(remap_s, 3),
        "qmm_s": round(qmm_s, 3) if profiled else None,
        "other_s": round(other_s, 3),
        "hit_rate": round(sc["hit_rate"], 3) if sc else None,
        "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 3),
    }


def print_breakdown(r: dict):
    wall = r["wall_s"]
    def pct(s):
        return f"{s/wall*100:5.1f}%" if wall > 0 and s is not None else "    —  "

    print(f"\n  [{r['label']}]  {r['tok_s']:.3f} tok/s  "
          f"{r['n_tokens']} tokens  {wall:.2f}s wall  {r['ssd_gb']:.1f} GB SSD"
          + (f"  hit={r['hit_rate']:.1%}" if r["hit_rate"] else ""))
    print(f"  {'Bucket':<18} {'seconds':>8}  {'% wall':>7}  {'ms/tok':>8}")
    print(f"  {'-'*46}")

    rows = [
        ("read (SSD)",    r["read_s"]),
        ("convert",       r["convert_s"]),
        ("remap",         r["remap_s"]),
    ]
    if r["profiled"]:
        rows.append(("compute (qmm+swig)", r["qmm_s"]))
    rows.append(("other", r["other_s"]))

    for name, s in rows:
        if s is None:
            continue
        ms_per_tok = s / r["n_tokens"] * 1000 if r["n_tokens"] else 0
        print(f"  {name:<18} {s:>8.3f}s  {pct(s)}  {ms_per_tok:>7.2f}ms")

    if r["profiled"]:
        print(f"\n  NOTE: compute times require forced mx.eval — tok/s is slower than reality.")
    else:
        print(f"\n  NOTE: 'other' includes compute + attention + non-MoE layers + Python overhead.")
        print(f"        Run --profiled (pass 2) to split compute out of 'other'.")


def main():
    args = parse_args()

    print("Loading model…")
    model, tokenizer, expert_store, config = build_streamed_model(
        model_path=Path(args.model),
        index_path=Path(args.index),
        top_k=args.k,
        native_reader_path=Path(args.native_reader),
    )
    q = config.get("quantization") or {}

    session_cache = None
    if args.cache:
        session_cache = SessionWindowNativeCache(
            max_bytes=2 * 1024**3, window_tokens=args.window_tokens
        )

    patch_switches(model, expert_store, fused=args.fused, session_cache=session_cache,
                   quantization=q, top_k=args.k)

    prompt_tokens = prompt_tokens_from_messages(
        tokenizer, [{"role": "user", "content": PROMPT}], enable_thinking=False,
    )
    label = f"K={args.k}" + (" fused" if args.fused else "") + (f" cache-H{args.window_tokens}" if args.cache else "")
    print(f"Prompt: {len(prompt_tokens)} tokens  {label}  max_decode={args.max_tokens}\n")

    # Pass 1: realistic (no forced sync)
    print("Pass 1 — realistic (lazy eval) …")
    r1 = run_pass(model, tokenizer, expert_store, config, prompt_tokens,
                  session_cache=session_cache, top_k=args.k,
                  max_tokens=args.max_tokens, profiled=False, label=label + " [realistic]")
    print_breakdown(r1)

    # Pass 2: profiled (forced mx.eval per MoE sub-op)
    print("\nPass 2 — profiled (forced sync per compute op) …")
    if args.cache:
        session_cache2 = SessionWindowNativeCache(
            max_bytes=2 * 1024**3, window_tokens=args.window_tokens
        )
        patch_switches(model, expert_store, fused=args.fused, session_cache=session_cache2,
                       quantization=q, top_k=args.k)
    else:
        session_cache2 = None
        patch_switches(model, expert_store, fused=args.fused, session_cache=None,
                       quantization=q, top_k=args.k)

    r2 = run_pass(model, tokenizer, expert_store, config, prompt_tokens,
                  session_cache=session_cache2, top_k=args.k,
                  max_tokens=args.max_tokens, profiled=True, label=label + " [profiled]")
    print_breakdown(r2)

    print("\n" + "=" * 55)
    print("BOTTLENECK SUMMARY")
    print("=" * 55)
    wall = r1["wall_s"]
    read_frac  = r1["read_s"] / wall
    conv_frac  = r1["convert_s"] / wall
    # Use profiled pass for compute estimate, scaled to realistic wall time
    if r2["qmm_s"]:
        # compute fraction from profiled pass (ratio is more reliable than absolute)
        prof_moe_total = r2["read_s"] + r2["convert_s"] + r2["remap_s"] + r2["qmm_s"]
        compute_frac = (r2["qmm_s"] / r2["wall_s"]) if r2["wall_s"] > 0 else 0
    else:
        compute_frac = 0
    other_frac = max(0, 1 - read_frac - conv_frac - compute_frac)

    print(f"  Realistic decode: {r1['tok_s']:.3f} tok/s  ({wall:.2f}s / {r1['n_tokens']} tokens)")
    print()
    print(f"  {'Bucket':<28} {'% of wall':>10}  {'ms/tok':>8}")
    print(f"  {'-'*50}")
    for name, frac, ms in [
        ("SSD read",    read_frac,    r1["read_s"]    / r1["n_tokens"] * 1000),
        ("Convert",     conv_frac,    r1["convert_s"] / r1["n_tokens"] * 1000),
        ("Compute (MoE qmm+swig)", compute_frac, r2["qmm_s"] / r2["n_tokens"] * 1000 if r2["qmm_s"] else 0),
        ("Other (attn+non-MoE+Python)", other_frac, other_frac * wall / r1["n_tokens"] * 1000),
    ]:
        bar = "█" * int(frac * 30)
        print(f"  {name:<28} {frac*100:>8.1f}%  {ms:>7.2f}ms  {bar}")

    expert_store.close()


if __name__ == "__main__":
    main()
