#!/usr/bin/env python3
"""Benchmark decode throughput with optional expert caching and timing profiling.

Isolates the decode path from server overhead. Supports:
  - Expert cache strategies (none, session_window_native)
  - Per-component timing breakdown (--profile)
  - Two-pass profiling: realistic tok/s + forced-sync compute breakdown (--profile --two-pass)

Works with any model supported by build_streamed_model.

Examples:
  # Basic decode throughput
  poetry run python benchmarks/bench_decode.py \\
    --model <model_path> --index .run/qwen35b-expert-index.json \\
    --native-reader .run/libexpert_reader.dylib

  # With session-window expert cache
  poetry run python benchmarks/bench_decode.py \\
    --model <model_path> --index .run/qwen35b-expert-index.json \\
    --native-reader .run/libexpert_reader.dylib \\
    --expert-cache-strategy session_window_native --expert-window-tokens 2

  # Full timing breakdown (slows tok/s due to forced GPU sync)
  poetry run python benchmarks/bench_decode.py \\
    --model <model_path> --index .run/qwen35b-expert-index.json \\
    --native-reader .run/libexpert_reader.dylib --profile --two-pass
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lib.loader import parse_bytes, save_json  # noqa: E402

import mlx.core as mx  # noqa: E402
from mlx_lm.generate import generate_step  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from streaming_qwen.prefetch_switch import PrefetchManager  # noqa: E402
from streaming_qwen.runtime import (  # noqa: E402
    begin_session_cache_request,
    build_streamed_model,
    collect_session_cache_stats,
    collect_window_cache_stats,
    complete_session_cache_token,
    end_session_cache_request,
    set_routed_top_k,
    set_session_cache_phase,
    set_window_cache_enabled,
)
from streaming_qwen.server.protocol import prompt_tokens_from_messages  # noqa: E402
from streaming_qwen.streamed_switch import STREAM_STATS, reset_stream_stats, set_stream_profiling  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark decode throughput")
    p.add_argument("--model", required=True, help="Local model snapshot path")
    p.add_argument("--index", required=True, help="Path to expert_index.json")
    p.add_argument("--native-reader", required=True, help="Path to native expert reader dylib")
    p.add_argument("--prompt", default="Output the word hello exactly 160 times separated by spaces.",
                   help="Prompt text")
    p.add_argument("--max-tokens", type=int, default=256, help="Max decode tokens")
    p.add_argument("--routed-top-k", type=int, default=4, help="Routed expert top-k")
    p.add_argument("--prefill-step-size", type=int, default=1024, help="Prefill chunk size")
    p.add_argument("--component-workers", type=int, default=3, help="Native component workers")
    p.add_argument("--expert-cache-strategy", default="none",
                   choices=["none", "window_exact", "session_window_native"],
                   help="Expert cache strategy")
    p.add_argument("--expert-window-tokens", type=int, default=0, help="Cache window size in tokens")
    p.add_argument("--expert-cache-bytes", default="1G", help="Expert cache budget (e.g. 1G, 500M)")
    p.add_argument("--session-id", default="bench-session", help="Session id for cache benchmarks")
    p.add_argument("--use-prefetch", action="store_true", help="Enable expert prefetch")
    p.add_argument("--profile", action="store_true",
                   help="Print per-component timing breakdown (forces mx.eval per op in two-pass mode)")
    p.add_argument("--two-pass", action="store_true",
                   help="Run a second profiled pass with forced GPU sync to isolate compute time "
                        "(implies --profile; tok/s in pass 2 will be lower than production)")
    p.add_argument("--output", default=None, help="JSON output path")
    return p.parse_args()


def _prefill(model, prompt_tokens, prompt_cache, step_size):
    prompt = mx.array(prompt_tokens, dtype=mx.uint32)
    if prompt.size <= 1:
        return prompt
    remaining = prompt[:-1]
    while remaining.size > 0:
        n = min(step_size, remaining.size)
        model(remaining[:n][None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])
        remaining = remaining[n:]
        mx.clear_cache()
    return prompt[-1:]


def _tps_after_skip(times, started, skip):
    if len(times) <= skip:
        return 0.0
    begin = started if skip == 0 else times[skip - 1]
    elapsed = times[-1] - begin
    return (len(times) - skip) / elapsed if elapsed > 0 else 0.0


MOE_STAT_KEYS = [
    ("GPU sync + remap", "remap_seconds"),
    ("SSD reads",        "load_seconds"),
    ("bytes→mx.array",   "convert_seconds"),
    ("qmm_up  (GPU)",    "qmm_up_seconds"),
    ("qmm_gate (GPU)",   "qmm_gate_seconds"),
    ("swiglu  (GPU)",    "swiglu_seconds"),
    ("qmm_down (GPU)",   "qmm_down_seconds"),
]


def _run_decode(model, tokenizer, expert_store, *, prompt_tokens, args, cache_enabled,
                profiled: bool) -> dict:
    expert_store.reset_stats()
    reset_stream_stats()
    set_stream_profiling(profiled)
    mx.reset_peak_memory()

    prompt_cache = make_prompt_cache(model)
    begin_session_cache_request(model, session_id=args.session_id, phase="prefill",
                                enabled=cache_enabled, ephemeral=False)
    set_window_cache_enabled(model, cache_enabled, reset=False)
    set_routed_top_k(model, args.routed_top_k)

    last = _prefill(model, prompt_tokens, prompt_cache, args.prefill_step_size)

    expert_store.reset_stats()
    reset_stream_stats()
    set_session_cache_phase(model, "decode")

    eos_ids = set(getattr(tokenizer, "eos_token_ids", []) or [])
    generated: list[int] = []
    times: list[float] = []
    started = time.perf_counter()
    for token, _ in generate_step(last, model, max_tokens=args.max_tokens,
                                  prompt_cache=prompt_cache,
                                  prefill_step_size=args.prefill_step_size):
        generated.append(int(token))
        times.append(time.perf_counter())
        complete_session_cache_token(model)
        if int(token) in eos_ids:
            break

    set_window_cache_enabled(model, False, reset=False)
    end_session_cache_request(model)
    set_stream_profiling(False)

    elapsed = (times[-1] - started) if times else 0.0
    n_tok = len(generated)

    result = {
        "profiled_gpu_ops": profiled,
        "prompt_tokens": len(prompt_tokens),
        "generated_tokens": n_tok,
        "tps_all": round(_tps_after_skip(times, started, 0), 3),
        "tps_skip2": round(_tps_after_skip(times, started, 2), 3),
        "tps_skip4": round(_tps_after_skip(times, started, 4), 3),
        "elapsed_s": round(elapsed, 3),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
        "expert_store": dict(expert_store.stats),
        "preview": tokenizer.decode(generated[:64]),
    }

    # Cache stats
    window_stats = collect_window_cache_stats(model)
    result["window_cache"] = {**window_stats,
                              "current_gib": round(window_stats["current_bytes"] / 1024**3, 3)}
    session_stats = collect_session_cache_stats(model)
    if session_stats:
        result["session_cache"] = {**session_stats,
                                   "current_gib": round(session_stats["current_bytes"] / 1024**3, 3)}

    # Prefetch stats
    prefetch_mgr: PrefetchManager | None = getattr(expert_store, "prefetch_manager", None)
    if prefetch_mgr is not None:
        ps = prefetch_mgr.stats
        total_pf = ps["prefetch_hits"] + ps["prefetch_misses"]
        result["prefetch"] = {**ps, "hit_rate": round(ps["prefetch_hits"] / total_pf, 3) if total_pf else 0.0}

    read_s = expert_store.stats["read_seconds"]
    result["expert_read_gbps"] = round(
        (expert_store.stats["bytes_read"] / 1024**3) / read_s if read_s > 0 else 0.0, 3
    )

    # Timing breakdown
    moe_total = sum(STREAM_STATS[k] for _, k in MOE_STAT_KEYS)
    non_moe = max(0.0, elapsed - moe_total)
    profile_rows = [
        {
            "component": label,
            "total_s": round(STREAM_STATS[key], 4),
            "per_token_ms": round(STREAM_STATS[key] / n_tok * 1000, 2) if n_tok else 0.0,
            "pct": round(STREAM_STATS[key] / elapsed * 100, 1) if elapsed else 0.0,
        }
        for label, key in MOE_STAT_KEYS
    ]
    profile_rows.append({
        "component": "non-MoE (inferred)",
        "total_s": round(non_moe, 4),
        "per_token_ms": round(non_moe / n_tok * 1000, 2) if n_tok else 0.0,
        "pct": round(non_moe / elapsed * 100, 1) if elapsed else 0.0,
    })
    result["timing_profile"] = {"n_tokens": n_tok, "rows": profile_rows}

    return result


def _print_breakdown(result: dict) -> None:
    rows = result["timing_profile"]["rows"]
    elapsed = result["elapsed_s"]
    n_tok = result["generated_tokens"]
    label = "profiled (GPU synced)" if result["profiled_gpu_ops"] else "realistic (lazy eval)"
    print(f"\n=== Timing breakdown [{label}] ===")
    print(f"  {'Component':<22} {'Total':>8}  {'ms/tok':>7}  {'%':>6}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*7}  {'-'*6}")
    for row in rows:
        print(f"  {row['component']:<22} {row['total_s']:>8.3f}s"
              f"  {row['per_token_ms']:>6.1f}ms  {row['pct']:>5.1f}%")
    if elapsed and n_tok:
        print(f"  {'TOTAL':<22} {elapsed:>8.3f}s  {elapsed/n_tok*1000:>6.1f}ms  {'100.0':>5}%")
    if result["profiled_gpu_ops"]:
        print("  NOTE: --two-pass forces mx.eval per GPU op → tok/s lower than production.")
    else:
        print("  NOTE: qmm/swiglu times are 0 in realistic pass (lazy eval). Add --two-pass for compute breakdown.")


def main() -> None:
    args = parse_args()
    cache_enabled = args.expert_cache_strategy != "none" and args.expert_window_tokens > 0

    model, tokenizer, expert_store, _ = build_streamed_model(
        model_path=Path(args.model),
        index_path=Path(args.index),
        top_k=args.routed_top_k,
        native_reader_path=Path(args.native_reader),
        component_workers=args.component_workers,
        expert_cache_strategy=args.expert_cache_strategy,
        expert_window_tokens=args.expert_window_tokens,
        expert_cache_bytes=parse_bytes(args.expert_cache_bytes),
        use_prefetch=args.use_prefetch,
    )
    try:
        prompt_tokens = prompt_tokens_from_messages(
            tokenizer, [{"role": "user", "content": args.prompt}], enable_thinking=False
        )

        # Pass 1: realistic (lazy eval)
        result = _run_decode(model, tokenizer, expert_store,
                             prompt_tokens=prompt_tokens, args=args,
                             cache_enabled=cache_enabled, profiled=False)

        print(f"top_k={args.routed_top_k}  tps={result['tps_skip2']:.2f}  "
              f"peak_mem={result['peak_memory_gb']:.2f}GB  "
              f"expert_gbps={result['expert_read_gbps']:.2f}")
        if args.profile or args.two_pass:
            _print_breakdown(result)

        # Pass 2: profiled (forced GPU sync per MoE op) — optional
        result2 = None
        if args.two_pass:
            print("\nPass 2 — profiled (forced GPU sync) …")
            result2 = _run_decode(model, tokenizer, expert_store,
                                  prompt_tokens=prompt_tokens, args=args,
                                  cache_enabled=cache_enabled, profiled=True)
            _print_breakdown(result2)

        output = {"pass1_realistic": result}
        if result2:
            output["pass2_profiled"] = result2
        save_json(output, args.output)
    finally:
        expert_store.close()


if __name__ == "__main__":
    main()
