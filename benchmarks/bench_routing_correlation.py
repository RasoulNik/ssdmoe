#!/usr/bin/env python3
"""Measure expert routing correlation between Qwen3.5-35B and Qwen3.5-122B.

Both models share the same 256-expert index space (K=8). This script runs
both models on identical prompts and computes, per layer per token, how
much their expert selections overlap.

If overlap is high (>50%), the 35B model can serve as a draft router —
running 1 token ahead and prefetching the likely 122B experts before the
122B model needs them, hiding the SSD read latency.

Layer alignment strategy (STRUCTURAL, not proportional):
  Both models share identical group structure: [linear, linear, linear, full_attention] × N
    35B:  10 groups × 4 layers = 40 layers  → groups 0–9  (absolute layers 0–39)
    122B: 12 groups × 4 layers = 48 layers  → groups 0–11 (absolute layers 0–47)
  Mapping: 35B layer i → 122B layer i  (identity, for i=0..39)
  Unmatched: 122B layers 40–47 have no 35B counterpart (tail groups 10–11)

Usage:
  poetry run python benchmarks/bench_routing_correlation.py \\
    --model-large  ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-122B-A10B-4bit/snapshots/... \\
    --index-large  .run/qwen35-122b-expert-index.json \\
    --model-small  ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/... \\
    --index-small  .run/qwen35-35b-expert-index.json \\
    --native-reader .run/libexpert_reader.dylib \\
    --tokens 32
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

from streaming_moe.runtime import build_streamed_model, set_routed_top_k
from streaming_moe.streamed_switch import StreamedSwitchGLU
from streaming_moe.prefetch_switch import PrefetchingStreamedSwitchGLU


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-large", required=True, help="Path to 122B model snapshot")
    p.add_argument("--index-large", required=True)
    p.add_argument("--model-small", required=True, help="Path to 35B model snapshot")
    p.add_argument("--index-small", required=True)
    p.add_argument("--native-reader", required=True)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--routed-top-k", type=int, default=8)
    p.add_argument("--component-workers", type=int, default=3)
    p.add_argument("--prompt", default="Explain how neural networks learn from data.")
    p.add_argument("--output", default=None, help="JSON output path")
    return p.parse_args()


# ── Recording hook ────────────────────────────────────────────────────────────

class _RecordingSwitch(nn.Module):
    """Thin wrapper recording expert selections, then delegating to real switch."""
    def __init__(self, inner, layer_idx: int, record_dict: dict):
        super().__init__()
        self._inner = inner
        self._layer_idx = layer_idx
        self._record = record_dict  # layer_idx → list of frozenset

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        indices_np = np.array(indices.tolist(), dtype=np.int32)
        selected = frozenset(np.unique(indices_np).tolist())
        self._record[self._layer_idx].append(selected)
        return self._inner(x, indices)


def install_hooks(model, record_dict: dict) -> list[int]:
    from streaming_moe.runtime import _get_moe_module
    text_model = getattr(getattr(model, "language_model", model), "model", model)
    patched = []
    for i, layer in enumerate(text_model.layers):
        moe_module, _ = _get_moe_module(layer)
        if moe_module is None:
            continue
        switch = getattr(moe_module, "switch_mlp", None)
        if switch is None:
            continue
        if isinstance(switch, (StreamedSwitchGLU, PrefetchingStreamedSwitchGLU)):
            moe_module.switch_mlp = _RecordingSwitch(switch, i, record_dict)
            patched.append(i)
    return patched


def run_decode(model, tokenizer, prompt: str, n_tokens: int, top_k: int) -> list[int]:
    """Prefill + decode n_tokens, return generated token ids."""
    from mlx_lm.generate import generate_step
    set_routed_top_k(model, top_k)
    tokens = tokenizer.encode(prompt)
    prompt_arr = mx.array(tokens, dtype=mx.uint32)
    cache = make_prompt_cache(model)
    if prompt_arr.size > 1:
        model(prompt_arr[:-1][None], cache=cache)
        mx.eval([c.state for c in cache])
    last = prompt_arr[-1:]
    eos_ids = set(getattr(tokenizer, "eos_token_ids", []) or [])
    generated = []
    for token, _ in generate_step(last, model, max_tokens=n_tokens, prompt_cache=cache):
        mx.eval(token)
        tid = int(token)
        generated.append(tid)
        if tid in eos_ids:
            break
    return generated


def main():
    args = parse_args()

    # ── Load both models ──
    print("Loading 122B model …")
    large_selections: dict[int, list[frozenset]] = defaultdict(list)
    model_large, tok_large, store_large, _ = build_streamed_model(
        Path(args.model_large), Path(args.index_large),
        native_reader_path=Path(args.native_reader),
        component_workers=args.component_workers,
    )
    large_layers = install_hooks(model_large, large_selections)
    print(f"  Hooked {len(large_layers)} MoE layers.")

    print("Loading 35B model …")
    small_selections: dict[int, list[frozenset]] = defaultdict(list)
    model_small, tok_small, store_small, _ = build_streamed_model(
        Path(args.model_small), Path(args.index_small),
        native_reader_path=Path(args.native_reader),
        component_workers=args.component_workers,
    )
    small_layers = install_hooks(model_small, small_selections)
    print(f"  Hooked {len(small_layers)} MoE layers.")

    # ── Run both models on same prompt ──
    # Run small model first (faster), then large model
    print(f"\nRunning 35B on {args.tokens} decode tokens …")
    small_tokens = run_decode(model_small, tok_small, args.prompt, args.tokens, args.routed_top_k)
    print(f"  Generated: {tok_small.decode(small_tokens[:16])} …")

    print(f"\nRunning 122B on {args.tokens} decode tokens …")
    large_tokens = run_decode(model_large, tok_large, args.prompt, args.tokens, args.routed_top_k)
    print(f"  Generated: {tok_large.decode(large_tokens[:16])} …")

    # ── Align layers (STRUCTURAL mapping) ──
    # Both models share the same group pattern: [linear, linear, linear, full_attn] × N
    #   35B:  10 groups = 40 layers (0–39)
    #   122B: 12 groups = 48 layers (0–47)
    # Correct mapping: 35B layer i → 122B layer i  (identity for i=0..39)
    # 122B layers 40–47 are unmatched (tail groups 10–11 have no 35B counterpart).
    n_large = len(large_layers)
    n_small = len(small_layers)

    # Build structural layer alignment map: identity where possible
    # small_layers and large_layers are sorted lists of MoE layer indices
    # Since all layers are MoE in both models, small_layers = [0..39], large_layers = [0..47]
    large_layer_set = set(large_layers)
    layer_map: list[tuple[int, int]] = []
    unmatched_large: list[int] = []
    for sl in small_layers:
        if sl in large_layer_set:
            layer_map.append((sl, sl))   # same absolute index
        # (if 35B had a layer not in 122B, we'd skip — shouldn't happen here)
    unmatched_large = [ll for ll in large_layers if ll not in {sl for sl, _ in layer_map}]

    # ── Compute routing correlation ──
    # For each aligned layer pair, compute per-token Jaccard between
    # small model's selection and large model's selection.
    print("\nComputing routing correlation …")

    per_layer_jaccard = []  # list of (small_layer, large_layer, mean_jaccard, n_tokens)
    all_jaccards = []

    for sl, ll in layer_map:
        s_sel = small_selections[sl]
        l_sel = large_selections[ll]
        n = min(len(s_sel), len(l_sel))
        if n < 2:
            continue
        jaccards = []
        for t in range(n):
            a, b = s_sel[t], l_sel[t]
            if not a and not b:
                jaccards.append(1.0)
            elif not (a | b):
                jaccards.append(0.0)
            else:
                jaccards.append(len(a & b) / len(a | b))
        mean_j = float(np.mean(jaccards))
        per_layer_jaccard.append((sl, ll, mean_j, n))
        all_jaccards.extend(jaccards)

    # ── Zero-shot hit rate ──
    # Treat small model selection as a PREDICTION of large model selection.
    # Hit rate = |small ∩ large| / |large|  (fraction of large experts that small predicted)
    hit_rates = []
    for sl, ll in layer_map:
        s_sel = small_selections[sl]
        l_sel = large_selections[ll]
        n = min(len(s_sel), len(l_sel))
        for t in range(n):
            if l_sel[t]:
                hr = len(s_sel[t] & l_sel[t]) / len(l_sel[t])
                hit_rates.append(hr)

    # ── Print results ──
    print("\n" + "═" * 65)
    print("Cross-model expert routing correlation (35B → 122B)")
    print("  Mapping: STRUCTURAL (35B layer i → 122B layer i, identity)")
    print("─" * 65)
    print(f"  Aligned layer pairs:         {len(per_layer_jaccard)}  (35B layers 0–{small_layers[-1]})")
    print(f"  Unmatched 122B tail layers:  {len(unmatched_large)}  (layers {unmatched_large[0]}–{unmatched_large[-1]} → no 35B coverage)")
    print(f"  Tokens compared per layer:   ~{per_layer_jaccard[0][3] if per_layer_jaccard else 0}")
    print()
    print(f"  Jaccard similarity (set overlap):")
    print(f"    Mean:   {np.mean(all_jaccards)*100:5.1f}%")
    print(f"    Median: {np.median(all_jaccards)*100:5.1f}%")
    print(f"    Min:    {np.min(all_jaccards)*100:5.1f}%")
    print(f"    Max:    {np.max(all_jaccards)*100:5.1f}%")
    print()
    print(f"  Zero-shot hit rate (35B predicts 122B):")
    print(f"    Mean:   {np.mean(hit_rates)*100:5.1f}%  — fraction of 122B experts correctly predicted")
    print(f"    Median: {np.median(hit_rates)*100:5.1f}%")

    # ── Per-layer table ──
    print()
    print("Per-layer Jaccard (sample, every 4th aligned pair):")
    print(f"  {'35B layer':>10}  {'122B layer':>10}  {'Jaccard':>8}  {'n_tok':>6}")
    print("  " + "─" * 40)
    for i, (sl, ll, mj, n) in enumerate(per_layer_jaccard[::4]):
        print(f"  {sl:>10}  {ll:>10}  {mj*100:>7.1f}%  {n:>6}")

    # ── SSD savings estimate ──
    mean_hit = float(np.mean(hit_rates))
    sample_layer = str(large_layers[0])
    layer_info = store_large.expert_reads[sample_layer]
    bytes_per_expert = sum(info.get("expert_size", 0) for info in layer_info.values())
    gb_per_token = (n_large * args.routed_top_k * bytes_per_expert) / 1e9

    print()
    print("─" * 65)
    print(f"  Expert size (122B): {bytes_per_expert/1e6:.2f} MB")
    print(f"  SSD/token baseline: {gb_per_token*1000:.0f} MB")
    effective_gb = gb_per_token * (1 - mean_hit)
    ms_with = effective_gb / 3.4 * 1000
    print(f"  With {mean_hit*100:.0f}% hit rate (perfect prefetch): {effective_gb*1000:.0f} MB/tok → {ms_with:.0f} ms → ~{1000/ms_with:.2f} tok/s")
    print()
    if mean_hit > 0.6:
        print("  ✓ STRONG correlation — 35B can serve as effective draft router")
    elif mean_hit > 0.35:
        print("  ~ MODERATE correlation — worth training a lightweight projector")
    else:
        print("  ✗ WEAK correlation — cross-model routing not viable zero-shot")

    # ── Save results ──
    results = {
        "large_model": args.model_large,
        "small_model": args.model_small,
        "n_tokens": args.tokens,
        "top_k": args.routed_top_k,
        "large_moe_layers": n_large,
        "small_moe_layers": n_small,
        "layer_alignment": "structural_identity",
        "aligned_pairs": len(layer_map),
        "unmatched_large_layers": unmatched_large,
        "jaccard": {
            "mean": float(np.mean(all_jaccards)),
            "median": float(np.median(all_jaccards)),
            "min": float(np.min(all_jaccards)),
            "max": float(np.max(all_jaccards)),
        },
        "hit_rate": {
            "mean": float(np.mean(hit_rates)),
            "median": float(np.median(hit_rates)),
        },
        "per_layer": [
            {"small_layer": sl, "large_layer": ll, "jaccard": mj, "n_tokens": n}
            for sl, ll, mj, n in per_layer_jaccard
        ],
    }

    out = args.output or "benchmarks/results/routing-correlation.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
