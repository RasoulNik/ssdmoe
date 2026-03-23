#!/usr/bin/env python3
"""Benchmark decode throughput with optional rolling expert window.

This isolates the decode path from server overhead and reports throughput after
skipping the first N generated tokens so cold-fill does not dominate the result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

from streaming_qwen.runtime import (
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
from streaming_qwen.server.protocol import prompt_tokens_from_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark decode window cache")
    parser.add_argument("--model", required=True, help="Local model snapshot path")
    parser.add_argument("--index", required=True, help="Path to expert_index.json")
    parser.add_argument(
        "--prompt",
        default="Output the word hello exactly 160 times separated by spaces, and nothing else.",
        help="Prompt text",
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="Generation length")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Use thinking-enabled chat template tokens",
    )
    parser.add_argument("--routed-top-k", type=int, default=4, help="Top-k during both prefill and decode")
    parser.add_argument(
        "--expert-cache-strategy",
        default="none",
        choices=["none", "window_exact", "session_window_native"],
        help="Decode-time expert cache strategy",
    )
    parser.add_argument(
        "--expert-window-tokens",
        type=int,
        default=0,
        help="Rolling expert window size in tokens",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=1024,
        help="Prompt prefill chunk size",
    )
    parser.add_argument(
        "--native-reader",
        required=True,
        help="Path to native expert reader dylib",
    )
    parser.add_argument(
        "--component-workers",
        type=int,
        default=3,
        help="Concurrent native component readers",
    )
    parser.add_argument(
        "--expert-cache-bytes",
        default="1G",
        help="Global session expert-cache budget",
    )
    parser.add_argument(
        "--session-id",
        default="bench-session",
        help="Explicit session id to use for session-scoped cache benchmarks",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path",
    )
    return parser.parse_args()


def prefill_prompt(model, prompt_tokens: list[int], prompt_cache, step_size: int) -> mx.array:
    prompt = mx.array(prompt_tokens, dtype=mx.uint32)
    if prompt.size <= 1:
        return prompt
    remaining = prompt[:-1]
    while remaining.size > 0:
        n_to_process = min(step_size, remaining.size)
        model(remaining[:n_to_process][None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])
        remaining = remaining[n_to_process:]
        mx.clear_cache()
    return prompt[-1:]


def throughput_after(times: list[float], start: float, skip: int) -> float:
    if len(times) <= skip:
        return 0.0
    begin = start if skip == 0 else times[skip - 1]
    elapsed = times[-1] - begin
    tokens = len(times) - skip
    return tokens / elapsed if elapsed > 0 else 0.0


def main() -> None:
    args = parse_args()
    cache_sizes = {"M": 1_000_000, "G": 1_000_000_000, "MB": 1_000_000, "GB": 1_000_000_000, "": 1}
    split = 0
    for ch in args.expert_cache_bytes:
        if not (ch.isdigit() or ch == "."):
            break
        split += 1
    digits = float(args.expert_cache_bytes[:split])
    suffix = args.expert_cache_bytes[split:].strip().upper()
    model, tokenizer, expert_store, _ = build_streamed_model(
        model_path=Path(args.model),
        index_path=Path(args.index),
        top_k=args.routed_top_k,
        native_reader_path=Path(args.native_reader),
        component_workers=args.component_workers,
        expert_cache_strategy=args.expert_cache_strategy,
        expert_window_tokens=args.expert_window_tokens,
        expert_cache_bytes=int(digits * cache_sizes[suffix]),
    )
    try:
        prompt_tokens = prompt_tokens_from_messages(
            tokenizer,
            [{"role": "user", "content": args.prompt}],
            enable_thinking=args.enable_thinking,
        )
        prompt_cache = make_prompt_cache(model)
        cache_enabled = args.expert_cache_strategy != "none" and args.expert_window_tokens > 0
        begin_session_cache_request(
            model,
            session_id=args.session_id,
            phase="prefill",
            enabled=cache_enabled,
            ephemeral=False,
        )
        set_window_cache_enabled(model, cache_enabled, reset=False)
        set_routed_top_k(model, args.routed_top_k)
        last_prompt_token = prefill_prompt(
            model,
            prompt_tokens,
            prompt_cache,
            args.prefill_step_size,
        )

        expert_store.reset_stats()
        mx.reset_peak_memory()
        set_routed_top_k(model, args.routed_top_k)
        set_session_cache_phase(model, "decode")
        set_window_cache_enabled(model, cache_enabled, reset=False)

        eos_ids = set(getattr(tokenizer, "eos_token_ids", []) or [])
        generated_tokens: list[int] = []
        token_times: list[float] = []
        started = time.perf_counter()
        for token, _ in generate_step(
            last_prompt_token,
            model,
            max_tokens=args.max_tokens,
            prompt_cache=prompt_cache,
            prefill_step_size=args.prefill_step_size,
        ):
            token = int(token)
            generated_tokens.append(token)
            token_times.append(time.perf_counter())
            complete_session_cache_token(model)
            if token in eos_ids:
                break

        result = {
            "prompt_tokens": len(prompt_tokens),
            "generated_tokens": len(generated_tokens),
            "completion_tps_all": round(throughput_after(token_times, started, 0), 3),
            "completion_tps_skip2": round(throughput_after(token_times, started, 2), 3),
            "completion_tps_skip3": round(throughput_after(token_times, started, 3), 3),
            "completion_tps_skip4": round(throughput_after(token_times, started, 4), 3),
            "elapsed_s": round((token_times[-1] - started) if token_times else 0.0, 3),
            "expert_store": dict(expert_store.stats),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
            "expert_cache_strategy": args.expert_cache_strategy,
            "expert_window_tokens": args.expert_window_tokens,
            "routed_top_k": args.routed_top_k,
            "preview": tokenizer.decode(generated_tokens[:64]),
        }
        window_stats = collect_window_cache_stats(model)
        result["window_cache"] = {
            **window_stats,
            "current_gib": round(window_stats["current_bytes"] / 1024**3, 3),
            "peak_gib": round(window_stats["peak_bytes"] / 1024**3, 3),
        }
        session_cache_stats = collect_session_cache_stats(model)
        if session_cache_stats:
            result["session_cache"] = {
                **session_cache_stats,
                "current_gib": round(session_cache_stats["current_bytes"] / 1024**3, 3),
                "peak_gib": round(session_cache_stats["peak_bytes"] / 1024**3, 3),
            }
        if expert_store.stats["read_seconds"] > 0:
            result["expert_read_gbps"] = round(
                (expert_store.stats["bytes_read"] / 1024**3) / expert_store.stats["read_seconds"],
                3,
            )
        else:
            result["expert_read_gbps"] = 0.0
    finally:
        set_window_cache_enabled(model, False, reset=False)
        end_session_cache_request(model)
        expert_store.close()

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
