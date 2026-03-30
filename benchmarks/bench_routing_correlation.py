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
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.loader import ensure_src_path
ensure_src_path()

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

from lib.hooks import install_recording_hooks
from lib.decode import run_decode
from lib.report import BenchReport, Table, Row
from streaming_moe.runtime import build_streamed_model, set_routed_top_k


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
    p.add_argument("--ssd-gbps", type=float, default=3.4, help="Measured SSD read bandwidth in GB/s")
    return p.parse_args()




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
    large_layers = install_recording_hooks(model_large, large_selections)
    print(f"  Hooked {len(large_layers)} MoE layers.")

    print("Loading 35B model …")
    small_selections: dict[int, list[frozenset]] = defaultdict(list)
    model_small, tok_small, store_small, _ = build_streamed_model(
        Path(args.model_small), Path(args.index_small),
        native_reader_path=Path(args.native_reader),
        component_workers=args.component_workers,
    )
    small_layers = install_recording_hooks(model_small, small_selections)
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

    # ── Compute SSD savings ──
    mean_hit = float(np.mean(hit_rates))
    sample_layer = str(large_layers[0])
    layer_info = store_large.expert_reads[sample_layer]
    bytes_per_expert = sum(info.get("expert_size", 0) for info in layer_info.values())
    gb_per_token = (n_large * args.routed_top_k * bytes_per_expert) / 1e9
    effective_gb = gb_per_token * (1 - mean_hit)
    ms_with = effective_gb / args.ssd_gbps * 1000
    ms_base = gb_per_token / args.ssd_gbps * 1000
    if mean_hit > 0.6:
        verdict = "STRONG — 35B can serve as effective draft router"
    elif mean_hit > 0.35:
        verdict = "MODERATE — worth training a lightweight projector"
    else:
        verdict = "WEAK — cross-model routing not viable zero-shot"

    # ── Per-layer table rows ──
    layer_rows = [
        Row([str(sl), str(ll), f"{mj*100:.1f}%", str(n)])
        for sl, ll, mj, n in per_layer_jaccard[::4]
    ]

    results = {
        "large_model": args.model_large,
        "small_model": args.model_small,
        "n_tokens": args.tokens,
        "top_k": args.routed_top_k,
        "ssd_gbps": args.ssd_gbps,
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
        "ssd_savings": {
            "baseline_gb_per_tok": gb_per_token,
            "baseline_ms": ms_base,
            "with_prefetch_gb_per_tok": effective_gb,
            "with_prefetch_ms": ms_with,
            "with_prefetch_tps": 1000 / ms_with if ms_with > 0 else 0,
        },
        "per_layer": [
            {"small_layer": sl, "large_layer": ll, "jaccard": mj, "n_tokens": n}
            for sl, ll, mj, n in per_layer_jaccard
        ],
    }

    report = BenchReport(
        title="Cross-model expert routing correlation (35B → 122B)",
        subtitle="Mapping: STRUCTURAL  (35B layer i → 122B layer i, identity)",
        tables=[
            Table(
                title="Jaccard similarity (set overlap) + hit rate",
                headers=["Metric", "Mean%", "Median%", "Min%", "Max%"],
                rows=[
                    Row(["Jaccard",
                         f"{np.mean(all_jaccards)*100:.1f}%",
                         f"{np.median(all_jaccards)*100:.1f}%",
                         f"{np.min(all_jaccards)*100:.1f}%",
                         f"{np.max(all_jaccards)*100:.1f}%"]),
                    Row(["Hit rate",
                         f"{np.mean(hit_rates)*100:.1f}%",
                         f"{np.median(hit_rates)*100:.1f}%", "—", "—"],
                        highlight=(mean_hit > 0.35)),
                ],
            ),
            Table(
                title="Per-layer Jaccard (every 4th aligned pair)",
                headers=["35B layer", "122B layer", "Jaccard", "n_tok"],
                rows=layer_rows,
            ),
            Table(
                title=f"SSD prefetch savings  (SSD bandwidth = {args.ssd_gbps} GB/s)",
                headers=["Scenario", "MB/tok", "ms/tok", "tok/s"],
                rows=[
                    Row(["Baseline",
                         f"{gb_per_token*1000:.0f} MB",
                         f"{ms_base:.0f} ms",
                         f"{1000/ms_base:.2f}"]),
                    Row([f"With {mean_hit*100:.0f}% prefetch",
                         f"{effective_gb*1000:.0f} MB",
                         f"{ms_with:.0f} ms",
                         f"{1000/ms_with:.2f}" if ms_with > 0 else "∞"],
                        highlight=True),
                ],
            ),
        ],
        notes=[f"Verdict: {verdict}"],
        raw=results,
    )
    report.print_terminal()
    out = args.output or "benchmarks/results/routing-correlation.json"
    report.save_json(out)


if __name__ == "__main__":
    main()
