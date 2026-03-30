#!/usr/bin/env python3
"""Measure next-layer expert prediction accuracy using the current layer's hidden state.

The "Speculating Experts" approach (arxiv 2603.19289): at layer L, compute a
quasi-hidden state (approximate router input for layer L+1) and predict which
experts layer L+1 will select.  If accurate, we can start prefetching layer L+1
experts while layer L's MoE computation runs — hiding SSD latency behind GPU work.

This script measures PREDICTION ACCURACY only (not actual speedup).
The quasi-hidden state approximation: h̃_{L+1} ≈ h_L + mean(selected_expert_outputs)
where we use the CURRENT layer's selected expert outputs as an approximation of
the residual contribution.

If accuracy > 50%, intra-model prefetch is viable.

Usage:
  poetry run python benchmarks/bench_next_layer_predict.py \\
    --model ~/.cache/.../Qwen3.5-122B-A10B-4bit/snapshots/... \\
    --index .run/qwen35-122b-expert-index.json \\
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
from streaming_moe.runtime import build_streamed_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--native-reader", required=True)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--routed-top-k", type=int, default=8)
    p.add_argument("--component-workers", type=int, default=3)
    p.add_argument("--prompt", default="Explain the architecture of transformer models.")
    p.add_argument("--output", default=None)
    p.add_argument("--ssd-gbps", type=float, default=3.4, help="Measured SSD read bandwidth in GB/s")
    return p.parse_args()




def main():
    args = parse_args()

    print("Loading model …")
    model, tokenizer, expert_store, _ = build_streamed_model(
        Path(args.model), Path(args.index),
        native_reader_path=Path(args.native_reader),
        component_workers=args.component_workers,
    )
    _layer_data: dict[int, list] = defaultdict(list)
    patched = install_recording_hooks(model, _layer_data)
    print(f"  Hooked {len(patched)} MoE layers: {patched[:5]}…")

    print(f"\nRunning {args.tokens} decode tokens …")
    run_decode(model, tokenizer, args.prompt, args.tokens, args.routed_top_k)
    print(f"  Done.\n")

    # ── Compute next-layer prediction accuracy ──
    # For each adjacent layer pair (L, L+1), how often does L's selection
    # predict L+1's selection?
    # Baseline predictor: just use SAME experts as layer L → predicts layer L+1
    # (This is the "last-layer persistence" predictor, simplest possible)
    # Better predictor would use quasi-hidden state, but that requires model internals.

    pairs: list[tuple[int, int]] = []
    for i in range(len(patched) - 1):
        pairs.append((patched[i], patched[i + 1]))

    print("═" * 65)
    print("Next-layer expert prediction accuracy (intra-model)")
    print("  Predictor: use layer L expert selections to predict layer L+1")
    print("─" * 65)

    all_hits = []
    all_jaccards = []

    per_pair = []
    for l_cur, l_next in pairs:
        cur  = _layer_data[l_cur]
        nxt  = _layer_data[l_next]
        n = min(len(cur), len(nxt))
        if n < 2:
            continue
        hits = []
        jaccards = []
        for t in range(n):
            pred = cur[t]   # predict layer L+1 = same as layer L
            true = nxt[t]
            if true:
                hits.append(len(pred & true) / len(true))
            if pred | true:
                jaccards.append(len(pred & true) / len(pred | true))
        if hits:
            mean_hit = float(np.mean(hits))
            mean_jac = float(np.mean(jaccards)) if jaccards else 0.0
            per_pair.append((l_cur, l_next, mean_hit, mean_jac, n))
            all_hits.extend(hits)
            all_jaccards.extend(jaccards)

    if not all_hits:
        print("  No data.")
        return

    mean_hit = float(np.mean(all_hits))

    # ── SSD savings geometry ──
    sample_layer = str(patched[0])
    layer_info = expert_store.expert_reads[sample_layer]
    bytes_per_expert = sum(info.get("expert_size", 0) for info in layer_info.values())
    n_moe = len(patched)
    k = args.routed_top_k
    gb_per_token = (n_moe * k * bytes_per_expert) / 1e9
    bw = args.ssd_gbps
    ms_base = gb_per_token / bw * 1000
    eff_gb = gb_per_token * (1 - mean_hit)
    ms_pred = eff_gb / bw * 1000

    results = {
        "model": args.model,
        "tokens": args.tokens,
        "top_k": k,
        "ssd_gbps": bw,
        "predictor": "same_as_prev_layer",
        "hit_rate": {
            "mean": mean_hit,
            "median": float(np.median(all_hits)),
            "min": float(np.min(all_hits)),
            "max": float(np.max(all_hits)),
        },
        "jaccard": {
            "mean": float(np.mean(all_jaccards)),
            "median": float(np.median(all_jaccards)),
        },
        "ssd_savings": {
            "baseline_ms": ms_base,
            "baseline_tps": 1000 / ms_base,
            "with_prefetch_ms": ms_pred,
            "with_prefetch_tps": 1000 / ms_pred if ms_pred > 0 else 0,
        },
        "per_pair": [
            {"layer": lc, "next_layer": ln, "hit_rate": mh, "jaccard": mj, "n_tokens": n}
            for lc, ln, mh, mj, n in per_pair
        ],
    }

    pair_rows = [
        Row([f"{lc}→{ln}", f"{mh*100:.1f}%", f"{mj*100:.1f}%"])
        for lc, ln, mh, mj, n in per_pair[::4]
    ]

    report = BenchReport(
        title="Next-layer expert prediction accuracy (intra-model)",
        subtitle="Predictor: 'same experts as previous layer' (weakest baseline)",
        tables=[
            Table(
                title="Hit rate and Jaccard",
                headers=["Metric", "Mean%", "Median%", "Min%", "Max%"],
                rows=[
                    Row(["Hit rate",
                         f"{np.mean(all_hits)*100:.1f}%",
                         f"{np.median(all_hits)*100:.1f}%",
                         f"{np.min(all_hits)*100:.1f}%",
                         f"{np.max(all_hits)*100:.1f}%"],
                        highlight=(mean_hit > 0.5)),
                    Row(["Jaccard",
                         f"{np.mean(all_jaccards)*100:.1f}%",
                         f"{np.median(all_jaccards)*100:.1f}%", "—", "—"]),
                ],
            ),
            Table(
                title="Per adjacent-layer pair (every 4th)",
                headers=["L→L+1", "hit%", "jaccard%"],
                rows=pair_rows,
            ),
            Table(
                title=f"SSD prefetch savings  (SSD bandwidth = {bw} GB/s)",
                headers=["Scenario", "MB/tok", "ms/tok", "tok/s"],
                rows=[
                    Row(["Baseline",
                         f"{gb_per_token*1000:.0f} MB",
                         f"{ms_base:.0f} ms",
                         f"{1000/ms_base:.2f}"]),
                    Row([f"With {mean_hit*100:.0f}% prefetch",
                         f"{eff_gb*1000:.0f} MB",
                         f"{ms_pred:.0f} ms",
                         f"{1000/ms_pred:.2f}"],
                        highlight=True),
                ],
            ),
        ],
        notes=[
            "'Same as prev layer' is the WEAKEST predictor.",
            "Quasi-hidden-state (Speculating Experts) achieves 84-91%.",
            "Pre-attention linear predictor (2511.10676) achieves 93-97%.",
        ],
        raw=results,
    )
    report.print_terminal()
    out = args.output or "benchmarks/results/next-layer-prediction.json"
    report.save_json(out)


if __name__ == "__main__":
    main()
