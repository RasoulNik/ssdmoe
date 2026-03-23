#!/usr/bin/env python3
"""Independent verification of three code changes.

Test A — Session-window cache (LUT + slab + copy_experts_multi)
  Compares: no cache  vs  session_window_native H=2
  Metric: tok/s, SSD GB read, hit_rate

Test B — Fused gate+up expert path (fused_expert.py from Codex tree)
  Compares: separate gate+up  vs  fused  vs  compiled-fused
  Metric: tok/s (pure compute difference, no I/O variation)

Test C — ExpertStore batch-read: native dispatch_apply vs Python ThreadPool
  Direct microbenchmark at the ExpertStore level, not end-to-end.
  Restores the ThreadPool path in a local function to compare fairly.
  Metric: GB/s for reading one layer's experts

Usage:
  PYTHONPATH=/path/to/MLX .venv/bin/python3 scripts/verify_changes.py \
      --model <model_path> --index <index_path> --native-reader <dylib_path>
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming_qwen.runtime import (
    build_streamed_model,
    iter_moe_layers,
    set_routed_top_k,
    begin_session_cache_request,
    end_session_cache_request,
    set_session_cache_phase,
    complete_session_cache_token,
    collect_session_cache_stats,
)
from streaming_qwen.session_window_cache import SessionWindowNativeCache
from streaming_qwen.streamed_switch import StreamedSwitchGLU, STREAM_STATS
from streaming_qwen.server.protocol import prompt_tokens_from_messages

# ---------------------------------------------------------------------------
PROMPT = "Output the word hello exactly 160 times separated by spaces, and nothing else."
MAX_TOKENS = 64
SKIP = 3   # skip first N tokens when computing throughput (warm-up)
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--native-reader", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--window-tokens", type=int, default=2)
    p.add_argument("--tests", default="A,B,C",
                   help="Comma-separated list of tests to run (A, B, C)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def tps_from_times(times: list[float], start: float, skip: int) -> float:
    if len(times) <= skip:
        return 0.0
    begin = start if skip == 0 else times[skip - 1]
    elapsed = times[-1] - begin
    n = len(times) - skip
    return n / elapsed if elapsed > 0 else 0.0


def run_decode(
    model,
    tokenizer,
    expert_store,
    *,
    prompt_tokens: list[int],
    max_tokens: int,
    session_cache: SessionWindowNativeCache | None,
    top_k: int,
    label: str,
) -> dict:
    """Run one full decode pass and return timing stats."""
    expert_store.reset_stats()
    for key in list(STREAM_STATS.keys()):
        STREAM_STATS[key] = 0

    prompt_cache = make_prompt_cache(model)
    set_routed_top_k(model, top_k)

    cache_enabled = session_cache is not None
    if cache_enabled:
        begin_session_cache_request(
            model, session_id="verify", phase="prefill",
            enabled=True, ephemeral=True,
        )
    # Prefill
    prompt = mx.array(prompt_tokens, dtype=mx.uint32)
    if prompt.size > 1:
        model(prompt[:-1][None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])
        mx.clear_cache()
    last = prompt[-1:]

    # Switch to decode
    expert_store.reset_stats()
    mx.reset_peak_memory()
    if cache_enabled:
        set_session_cache_phase(model, "decode")

    token_times: list[float] = []
    started = time.perf_counter()
    for token, _ in generate_step(
        last, model, max_tokens=max_tokens, prompt_cache=prompt_cache
    ):
        token_times.append(time.perf_counter())
        if cache_enabled:
            complete_session_cache_token(model)

    if cache_enabled:
        end_session_cache_request(model)

    result = {
        "label": label,
        "tok_s": round(tps_from_times(token_times, started, SKIP), 3),
        "elapsed_s": round((token_times[-1] - started) if token_times else 0.0, 2),
        "ssd_gb": round(expert_store.stats["bytes_read"] / 1e9, 2),
        "read_gbps": round(
            (expert_store.stats["bytes_read"] / 1e9) / expert_store.stats["read_seconds"]
            if expert_store.stats["read_seconds"] > 0 else 0.0, 2
        ),
        "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 3),
    }
    sc = collect_session_cache_stats(model)
    if sc:
        result["hit_rate"] = round(sc["hit_rate"], 3)
        result["hit_experts"] = sc["hit_experts"]
        result["miss_experts"] = sc["miss_experts"]
    return result


def patch_layers(model, expert_store, *, fused_gate_up: bool, compile_fused: bool,
                  session_cache, quantization: dict, top_k: int):
    """Replace switch_mlp on every MoE layer with the given settings."""
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
            fused_gate_up=fused_gate_up,
            compile_fused_gate_up=compile_fused,
            session_cache=session_cache,
        )
    set_routed_top_k(model, top_k)


# ---------------------------------------------------------------------------
# Test A: cache impact
# ---------------------------------------------------------------------------

def test_a(args, model, tokenizer, expert_store, config, prompt_tokens):
    print("\n" + "=" * 60)
    print("TEST A — Session-window cache impact")
    print("=" * 60)

    results = []

    # A1: no cache
    patch_layers(model, expert_store, fused_gate_up=False, compile_fused=False,
                 session_cache=None, quantization=config.get("quantization") or {}, top_k=args.k)
    r = run_decode(model, tokenizer, expert_store, prompt_tokens=prompt_tokens,
                   max_tokens=MAX_TOKENS, session_cache=None, top_k=args.k, label="no_cache")
    results.append(r)
    print(f"  no_cache:      {r['tok_s']:.3f} tok/s  {r['ssd_gb']:.2f} GB  {r['elapsed_s']:.1f}s")

    # A2: session_window_native H=window_tokens
    sc = SessionWindowNativeCache(max_bytes=2 * 1024**3, window_tokens=args.window_tokens)
    patch_layers(model, expert_store, fused_gate_up=False, compile_fused=False,
                 session_cache=sc, quantization=config.get("quantization") or {}, top_k=args.k)
    r = run_decode(model, tokenizer, expert_store, prompt_tokens=prompt_tokens,
                   max_tokens=MAX_TOKENS, session_cache=sc, top_k=args.k,
                   label=f"session_window_H{args.window_tokens}")
    results.append(r)
    print(f"  session_H{args.window_tokens}:     {r['tok_s']:.3f} tok/s  {r['ssd_gb']:.2f} GB  "
          f"{r['elapsed_s']:.1f}s  hit={r.get('hit_rate', 0):.1%}")

    b = results[0]["tok_s"]
    c = results[1]["tok_s"]
    print(f"\n  delta: {(c-b)/b*100:+.1f}%   cache/baseline: {c/b:.1%}")
    return results


# ---------------------------------------------------------------------------
# Test B: fused gate+up path
# ---------------------------------------------------------------------------

def test_b(args, model, tokenizer, expert_store, config, prompt_tokens):
    print("\n" + "=" * 60)
    print("TEST B — Fused gate+up expert path (no cache, pure compute)")
    print("=" * 60)

    q = config.get("quantization") or {}
    results = []
    # Note: fused_compiled is omitted — mx.compile(shapeless=True) cannot work
    # with gather_qmm because output shapes depend on rhs_indices at runtime.
    configs = [
        ("separate",  False, False),
        ("fused",     True,  False),
    ]
    for label, fused, compiled in configs:
        patch_layers(model, expert_store, fused_gate_up=fused, compile_fused=compiled,
                     session_cache=None, quantization=q, top_k=args.k)
        r = run_decode(model, tokenizer, expert_store, prompt_tokens=prompt_tokens,
                       max_tokens=MAX_TOKENS, session_cache=None, top_k=args.k, label=label)
        results.append(r)
        note = " (baseline)" if label == "separate" else ""
        print(f"  {label:<20} {r['tok_s']:.3f} tok/s  {r['elapsed_s']:.1f}s{note}")

    b = results[0]["tok_s"]
    for r in results[1:]:
        print(f"  {r['label']}: {(r['tok_s']-b)/b*100:+.1f}% vs separate")
    return results


# ---------------------------------------------------------------------------
# Test C: ExpertStore batch-read — ThreadPool vs native dispatch_apply
# ---------------------------------------------------------------------------

def test_c(args, model, tokenizer, expert_store, config, prompt_tokens):
    print("\n" + "=" * 60)
    print("TEST C — ExpertStore read_components_batched: ThreadPool vs native")
    print("=" * 60)

    # Find a layer with 3 components to read
    first_layer_key = next(iter(expert_store.expert_reads))
    layer_info = expert_store.expert_reads[first_layer_key]
    layer_idx = int(first_layer_key)
    components = list(layer_info.keys())
    # Use K experts for a realistic batch size
    expert_indices = list(range(args.k))
    REPS = 40

    # --- New path: native read_component_batches (current code) ---
    def native_batch():
        return expert_store.read_components_batched(layer_idx, expert_indices)

    # --- Old path: ThreadPool fan-out (restored locally, not in production code) ---
    nr = expert_store.native_reader

    def threadpool_batch():
        def _read(component):
            info = layer_info[component]
            fd = expert_store._fds[info["file"]]
            return component, nr.read_component_batch(
                fd=fd,
                abs_offset=info["abs_offset"],
                expert_stride=info["expert_stride"],
                expert_size=info["expert_size"],
                expert_indices=expert_indices,
            )
        with ThreadPoolExecutor(max_workers=len(components)) as pool:
            return dict(pool.map(_read, components))

    def bench_fn(fn, name):
        # warmup
        for _ in range(5):
            fn()
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        total_bytes = sum(
            layer_info[c]["expert_size"] * args.k for c in components
        )
        median_ms = statistics.median(times) * 1000
        p90_ms = sorted(times)[int(0.9 * REPS)] * 1000
        gbps = (total_bytes / 1e9) / statistics.median(times)
        print(f"  {name:<22} median={median_ms:.1f}ms  p90={p90_ms:.1f}ms  {gbps:.2f} GB/s  "
              f"({total_bytes/1e6:.1f} MB per call)")
        return {"name": name, "median_ms": round(median_ms, 2), "gbps": round(gbps, 2)}

    r_tp = bench_fn(threadpool_batch, "ThreadPool (old)")
    r_nb = bench_fn(native_batch,     "native_batch (new)")
    speedup = r_tp["median_ms"] / r_nb["median_ms"]
    print(f"\n  native_batch is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} "
          f"than ThreadPool (median latency)")
    return [r_tp, r_nb]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    tests = {t.strip().upper() for t in args.tests.split(",")}

    print("Loading model (this takes ~60s)…")
    model, tokenizer, expert_store, config = build_streamed_model(
        model_path=Path(args.model),
        index_path=Path(args.index),
        top_k=args.k,
        native_reader_path=Path(args.native_reader),
    )

    prompt_tokens = prompt_tokens_from_messages(
        tokenizer, [{"role": "user", "content": PROMPT}], enable_thinking=False,
    )
    print(f"Prompt: {len(prompt_tokens)} tokens  K={args.k}  "
          f"max_decode={MAX_TOKENS}  skip_first={SKIP}\n")

    results = {}
    if "A" in tests:
        results["A"] = test_a(args, model, tokenizer, expert_store, config, prompt_tokens)
    if "B" in tests:
        results["B"] = test_b(args, model, tokenizer, expert_store, config, prompt_tokens)
    if "C" in tests:
        results["C"] = test_c(args, model, tokenizer, expert_store, config, prompt_tokens)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "A" in results:
        a = results["A"]
        b, c = a[0]["tok_s"], a[1]["tok_s"]
        print(f"  Cache:          {b:.3f} → {c:.3f} tok/s  ({(c-b)/b*100:+.1f}%)  "
              f"hit={a[1].get('hit_rate', 0):.1%}")
    if "B" in results:
        b_res = results["B"]
        base = b_res[0]["tok_s"]
        for r in b_res:
            tag = " ←baseline" if r["label"] == "separate" else f"  ({(r['tok_s']-base)/base*100:+.1f}%)"
            print(f"  Fused {r['label']:<18} {r['tok_s']:.3f} tok/s{tag}")
    if "C" in results:
        c_res = results["C"]
        tp = next(r for r in c_res if "ThreadPool" in r["name"])
        nb = next(r for r in c_res if "native" in r["name"])
        print(f"  Batch read:     ThreadPool {tp['median_ms']:.1f}ms  "
              f"native {nb['median_ms']:.1f}ms  "
              f"({(tp['median_ms']-nb['median_ms'])/tp['median_ms']*100:+.1f}% faster)")

    expert_store.close()


if __name__ == "__main__":
    main()
