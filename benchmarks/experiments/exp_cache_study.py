#!/usr/bin/env python3
"""Study bounded expert reuse across consecutive decode tokens.

This script answers a narrower question than end-to-end serving:

- If we keep the routed experts from the last H tokens resident in memory,
  how much routed-expert traffic disappears?
- Does the realized SSD-side gain track the hit ratio, or does smaller miss
  batches collapse storage efficiency enough to erase the benefit?

The study works in two phases for each routed top-k:
1. Generate a real decode trace and record selected experts per layer per token.
2. Replay the resulting miss sets through the native reader with different
   sliding-window cache sizes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models import cache

from streaming_qwen.expert_store import ExpertStore
from streaming_qwen.runtime import build_streamed_model, iter_moe_layers
from streaming_qwen.streamed_switch import (
    get_expert_trace,
    reset_expert_trace,
    set_expert_tracing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study speculative expert reuse limits")
    parser.add_argument("--model", required=True, help="Local model snapshot path")
    parser.add_argument("--index", required=True, help="Path to expert_index.json")
    parser.add_argument(
        "--prompt",
        default=(
            "Explain why SSD streaming can bottleneck MoE inference on a laptop, "
            "and mention memory pressure too."
        ),
        help="Prompt text for the decode trace",
    )
    parser.add_argument(
        "--top-ks",
        default="2,4,8",
        help="Comma-separated routed top-k values to analyze",
    )
    parser.add_argument(
        "--windows",
        default="1,2,3",
        help="Comma-separated token-window sizes to simulate",
    )
    parser.add_argument("--max-tokens", type=int, default=24, help="Generation length")
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=1024,
        help="Prompt prefill chunk size",
    )
    parser.add_argument(
        "--native-reader",
        required=True,
        help="Path to native reader dylib",
    )
    parser.add_argument(
        "--component-workers",
        type=int,
        default=3,
        help="Concurrent native component readers",
    )
    parser.add_argument(
        "--output",
        default=".run/speculative-cache-study-20260321.json",
        help="JSON output path",
    )
    return parser.parse_args()


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    return tokenizer.encode(prompt)


def prefill_prompt(
    model,
    prompt_tokens: list[int],
    prompt_cache,
    prefill_step_size: int,
) -> mx.array:
    prompt = mx.array(prompt_tokens, dtype=mx.uint32)
    if prompt.size <= 1:
        return prompt

    remaining = prompt[:-1]
    while remaining.size > 0:
        n_to_process = min(prefill_step_size, remaining.size)
        model(remaining[:n_to_process][None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])
        remaining = remaining[n_to_process:]
        mx.clear_cache()
    return prompt[-1:]


def collect_decode_trace(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    prefill_step_size: int,
) -> tuple[list[list[set[int]]], dict]:
    prompt_tokens = encode_prompt(tokenizer, prompt)
    prompt_cache = cache.make_prompt_cache(model)
    last_prompt_token = prefill_prompt(
        model=model,
        prompt_tokens=prompt_tokens,
        prompt_cache=prompt_cache,
        prefill_step_size=prefill_step_size,
    )

    reset_expert_trace()
    set_expert_tracing(True)
    generated = []
    eos_ids = set(getattr(tokenizer, "eos_token_ids", []) or [])
    try:
        token_gen = generate_step(
            last_prompt_token,
            model,
            max_tokens=max_tokens,
            prompt_cache=prompt_cache,
            prefill_step_size=prefill_step_size,
        )
        for token, _logprobs in token_gen:
            generated.append(int(token))
            if int(token) in eos_ids:
                break
    finally:
        set_expert_tracing(False)

    trace = get_expert_trace()
    num_layers = len(list(iter_moe_layers(model)))
    if not trace:
        raise RuntimeError("No expert trace was collected")
    if len(trace) % num_layers != 0:
        raise RuntimeError(
            f"Trace length {len(trace)} is not divisible by num_layers {num_layers}"
        )

    steps: list[list[set[int]]] = []
    for i in range(0, len(trace), num_layers):
        chunk = trace[i : i + num_layers]
        steps.append([set(entry["selected_experts"]) for entry in chunk])

    return steps, {
        "prompt_tokens": len(prompt_tokens),
        "generated_tokens": len(generated),
        "generated_preview": tokenizer.decode(generated[:64]),
        "num_layers": num_layers,
    }


def expert_payload_bytes(index: dict) -> int:
    layer0 = index["expert_reads"]["0"]
    return sum(info["expert_size"] for info in layer0.values())


def simulate_window(
    steps: list[list[set[int]]],
    window_tokens: int,
    expert_bytes: int,
) -> dict:
    history: deque[list[set[int]]] = deque(maxlen=window_tokens)
    miss_steps: list[list[list[int]]] = []
    total_requested = 0
    total_hits = 0
    total_misses = 0
    steady_requested = 0
    steady_hits = 0
    steady_misses = 0
    resident_counts = []
    steady_resident_counts = []
    steady_miss_steps: list[list[list[int]]] = []

    for step_idx, layer_sets in enumerate(steps):
        cached_by_layer = []
        for layer_idx in range(len(layer_sets)):
            cached = set()
            for prev in history:
                cached.update(prev[layer_idx])
            cached_by_layer.append(cached)

        misses_for_step: list[list[int]] = []
        resident_count = 0
        for layer_idx, requested in enumerate(layer_sets):
            cached = cached_by_layer[layer_idx]
            hits = requested & cached
            misses = requested - cached
            total_requested += len(requested)
            total_hits += len(hits)
            total_misses += len(misses)
            misses_for_step.append(sorted(misses))
            resident_count += len(cached)
        resident_counts.append(resident_count)
        miss_steps.append(misses_for_step)
        if step_idx >= window_tokens:
            steady_requested += sum(len(requested) for requested in layer_sets)
            steady_hits += sum(len(layer_sets[layer_idx] & cached_by_layer[layer_idx]) for layer_idx in range(len(layer_sets)))
            steady_misses += sum(len(misses) for misses in misses_for_step)
            steady_resident_counts.append(resident_count)
            steady_miss_steps.append(misses_for_step)
        history.append(layer_sets)

    avg_resident = sum(resident_counts) / len(resident_counts)
    peak_resident = max(resident_counts)
    if steady_resident_counts:
        steady_avg_resident = sum(steady_resident_counts) / len(steady_resident_counts)
        steady_peak_resident = max(steady_resident_counts)
    else:
        steady_avg_resident = 0.0
        steady_peak_resident = 0
    return {
        "window_tokens": window_tokens,
        "total_requested_experts": total_requested,
        "total_hit_experts": total_hits,
        "total_miss_experts": total_misses,
        "hit_rate": (total_hits / total_requested) if total_requested else 0.0,
        "traffic_reduction": (total_hits / total_requested) if total_requested else 0.0,
        "avg_resident_experts": avg_resident,
        "peak_resident_experts": peak_resident,
        "avg_resident_gib": (avg_resident * expert_bytes) / 1024**3,
        "peak_resident_gib": (peak_resident * expert_bytes) / 1024**3,
        "miss_steps": miss_steps,
        "steady_state_start_step": window_tokens,
        "steady_state": {
            "requested_experts": steady_requested,
            "hit_experts": steady_hits,
            "miss_experts": steady_misses,
            "hit_rate": (steady_hits / steady_requested) if steady_requested else 0.0,
            "traffic_reduction": (steady_hits / steady_requested) if steady_requested else 0.0,
            "avg_resident_experts": steady_avg_resident,
            "peak_resident_experts": steady_peak_resident,
            "avg_resident_gib": (steady_avg_resident * expert_bytes) / 1024**3,
            "peak_resident_gib": (steady_peak_resident * expert_bytes) / 1024**3,
            "miss_steps": steady_miss_steps,
        },
    }


def replay_read_pattern(
    index_path: Path,
    native_reader: Path,
    component_workers: int,
    miss_steps: list[list[list[int]]],
) -> dict:
    with ExpertStore(
        index_path,
        native_reader_path=native_reader,
        component_workers=component_workers,
    ) as store:
        start = time.perf_counter()
        nonempty_reads = 0
        for layer_misses in miss_steps:
            for layer_idx, experts in enumerate(layer_misses):
                if not experts:
                    continue
                store.read_components_batched(layer_idx, experts)
                nonempty_reads += 1
        elapsed = time.perf_counter() - start
        stats = dict(store.stats)

    bytes_read = stats["bytes_read"]
    return {
        "elapsed_s": elapsed,
        "bytes_read": bytes_read,
        "gib_read": bytes_read / 1024**3,
        "gib_per_s": (bytes_read / 1024**3) / elapsed if elapsed else 0.0,
        "nonempty_layer_reads": nonempty_reads,
        "expert_store": stats,
    }


def main() -> None:
    args = parse_args()
    top_ks = [int(x) for x in args.top_ks.split(",") if x.strip()]
    windows = [int(x) for x in args.windows.split(",") if x.strip()]

    with Path(args.index).open() as f:
        index = json.load(f)
    expert_bytes = expert_payload_bytes(index)

    results = {
        "model": str(Path(args.model).expanduser().resolve()),
        "index": str(Path(args.index).expanduser().resolve()),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "prefill_step_size": args.prefill_step_size,
        "expert_payload_bytes": expert_bytes,
        "expert_payload_mib": expert_bytes / 1024**2,
        "top_k_runs": [],
    }

    for top_k in top_ks:
        mx.clear_cache()
        model, tokenizer, expert_store, _ = build_streamed_model(
            model_path=Path(args.model),
            index_path=Path(args.index),
            top_k=top_k,
            native_reader_path=Path(args.native_reader),
            component_workers=args.component_workers,
        )
        try:
            steps, trace_meta = collect_decode_trace(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                prefill_step_size=args.prefill_step_size,
            )
        finally:
            expert_store.close()

        baseline_miss_steps = [
            [sorted(layer_set) for layer_set in step]
            for step in steps
        ]
        baseline_replay = replay_read_pattern(
            index_path=Path(args.index),
            native_reader=Path(args.native_reader),
            component_workers=args.component_workers,
            miss_steps=baseline_miss_steps,
        )

        window_results = []
        for window_tokens in windows:
            sim = simulate_window(
                steps=steps,
                window_tokens=window_tokens,
                expert_bytes=expert_bytes,
            )
            replay = replay_read_pattern(
                index_path=Path(args.index),
                native_reader=Path(args.native_reader),
                component_workers=args.component_workers,
                miss_steps=sim["miss_steps"],
            )
            baseline_steady_replay = replay_read_pattern(
                index_path=Path(args.index),
                native_reader=Path(args.native_reader),
                component_workers=args.component_workers,
                miss_steps=baseline_miss_steps[window_tokens:],
            )
            steady_replay = replay_read_pattern(
                index_path=Path(args.index),
                native_reader=Path(args.native_reader),
                component_workers=args.component_workers,
                miss_steps=sim["steady_state"]["miss_steps"],
            )
            sim["replay"] = replay
            sim["realized_read_time_reduction"] = (
                1.0 - (replay["elapsed_s"] / baseline_replay["elapsed_s"])
                if baseline_replay["elapsed_s"]
                else 0.0
            )
            sim["realized_read_speedup"] = (
                baseline_replay["elapsed_s"] / replay["elapsed_s"]
                if replay["elapsed_s"]
                else 0.0
            )
            sim["steady_state"]["baseline_replay"] = baseline_steady_replay
            sim["steady_state"]["replay"] = steady_replay
            sim["steady_state"]["realized_read_time_reduction"] = (
                1.0 - (steady_replay["elapsed_s"] / baseline_steady_replay["elapsed_s"])
                if baseline_steady_replay["elapsed_s"]
                else 0.0
            )
            sim["steady_state"]["realized_read_speedup"] = (
                baseline_steady_replay["elapsed_s"] / steady_replay["elapsed_s"]
                if steady_replay["elapsed_s"]
                else 0.0
            )
            del sim["miss_steps"]
            del sim["steady_state"]["miss_steps"]
            window_results.append(sim)

        results["top_k_runs"].append(
            {
                "top_k": top_k,
                "trace": trace_meta,
                "baseline_replay": baseline_replay,
                "windows": window_results,
            }
        )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
