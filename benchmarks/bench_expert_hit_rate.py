#!/usr/bin/env python3
"""Measure cross-token expert selection hit rate per MoE layer.

For each pair of consecutive decode tokens, computes what fraction of layers
selected the *exact same* set of experts. Also prints the SSD read volume math.

Usage:
  poetry run python benchmarks/bench_expert_hit_rate.py \\
    --model ~/.cache/huggingface/hub/models--sjug--Nemotron-3-Super-120B-A12B-MLX-4bit/snapshots/ff505b4c07e1c23d8e650e9e37877bdf71c9424b \\
    --index .run/nemotron120b-expert-index.json \\
    --native-reader .run/libexpert_reader.dylib \\
    --tokens 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.loader import ensure_src_path
ensure_src_path()

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

from lib.hooks import install_recording_hooks
from lib.report import BenchReport, Table, Row
from streaming_moe.runtime import build_streamed_model, set_routed_top_k


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--native-reader", required=True)
    p.add_argument("--tokens", type=int, default=64, help="Decode tokens to observe")
    p.add_argument("--routed-top-k", type=int, default=8)
    p.add_argument("--component-workers", type=int, default=3)
    p.add_argument("--prompt", default="Explain how neural networks work in detail.")
    p.add_argument("--ssd-gbps", type=float, default=3.4, help="Measured SSD read bandwidth in GB/s")
    p.add_argument("--output", default=None, help="JSON output path")
    return p.parse_args()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"Loading model …")
    model, tokenizer, expert_store, config = build_streamed_model(
        Path(args.model),
        Path(args.index),
        native_reader_path=Path(args.native_reader),
        component_workers=args.component_workers,
    )
    set_routed_top_k(model, args.routed_top_k)
    _selections: dict[int, list] = defaultdict(list)
    patched_layers = install_recording_hooks(model, _selections)
    print(f"Hooked {len(patched_layers)} MoE layers: {patched_layers[:5]}…\n")

    # ── Prefill ──
    tokens = tokenizer.encode(args.prompt)
    prompt = mx.array(tokens, dtype=mx.uint32)
    prompt_cache = make_prompt_cache(model)
    if prompt.size > 1:
        model(prompt[:-1][None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])
    last = prompt[-1:]

    print(f"Prompt: {len(tokens)} tokens. Collecting {args.tokens} decode tokens …\n")

    # ── Decode loop ──
    from mlx_lm.generate import generate_step
    eos_ids = set(getattr(tokenizer, "eos_token_ids", []) or [])
    n = 0
    for token, _ in generate_step(last, model, max_tokens=args.tokens,
                                   prompt_cache=prompt_cache):
        mx.eval(token)
        n += 1
        if int(token) in eos_ids:
            break

    print(f"Collected {n} decode tokens across {len(patched_layers)} MoE layers.\n")

    if not any(_selections[l] for l in patched_layers):
        print("No data collected.")
        return

    # ── Window cache hit-rate analysis ──
    # For window H: cache = union of expert sets from the last H tokens at this layer.
    # Hit rate = |current ∩ cache| / |current|
    # "How many of the K experts I need now are already in RAM from the last H tokens?"
    WINDOWS = [1, 2]

    window_hits: dict[int, list[float]] = {H: [] for H in WINDOWS}
    window_hits_per_layer: dict[int, dict[int, list[float]]] = {H: {} for H in WINDOWS}

    for layer_idx in patched_layers:
        sel = _selections[layer_idx]
        for H in WINDOWS:
            fracs = []
            for t in range(H, len(sel)):
                current = sel[t]
                cache = frozenset().union(*sel[t - H: t])
                if current:
                    fracs.append(len(current & cache) / len(current))
            window_hits[H].extend(fracs)
            window_hits_per_layer[H][layer_idx] = fracs

    # ── Gather SSD geometry ──
    window_means: dict[int, float] = {}
    try:
        sample_layer = str(patched_layers[0])
        layer_info = expert_store.expert_reads[sample_layer]
        bytes_per_expert = sum(info.get("expert_size", 0) for info in layer_info.values())
        n_moe = len(patched_layers)
        k = args.routed_top_k
    except Exception:
        bytes_per_expert = 0; n_moe = 0; k = 0

    # ── Summary table rows ──
    summary_rows: list[Row] = []
    window_raw: dict = {}
    for H in WINDOWS:
        fracs = window_hits[H]
        if not fracs:
            continue
        mean_h = float(np.mean(fracs))
        med_h  = float(np.median(fracs))
        min_h  = float(np.min(fracs))
        max_h  = float(np.max(fracs))
        window_means[H] = mean_h
        cache_mb = H * n_moe * k * bytes_per_expert / 1e6 if bytes_per_expert else 0
        summary_rows.append(Row([
            f"H={H}",
            f"{mean_h*100:.1f}%",
            f"{med_h*100:.1f}%",
            f"{min_h*100:.1f}%",
            f"{max_h*100:.1f}%",
            f"{cache_mb:.0f} MB",
        ]))
        window_raw[H] = {"mean": mean_h, "median": med_h, "min": min_h, "max": max_h,
                         "cache_mb": cache_mb}

    # ── Per-layer breakdown rows ──
    layer_rows: list[Row] = []
    layer_raw: dict = {}
    for layer_idx in patched_layers[::4]:
        sel = _selections[layer_idx]
        h1 = window_hits_per_layer[1].get(layer_idx, [])
        h2 = window_hits_per_layer[2].get(layer_idx, [])
        n_unique = len(set(sel))
        r1 = f"{np.mean(h1)*100:.1f}%" if h1 else "—"
        r2 = f"{np.mean(h2)*100:.1f}%" if h2 else "—"
        layer_rows.append(Row([str(layer_idx), r1, r2, f"{n_unique}/{len(sel)}"]))
        layer_raw[layer_idx] = {"h1": float(np.mean(h1)) if h1 else None,
                                 "h2": float(np.mean(h2)) if h2 else None}

    # ── SSD savings rows ──
    ssd_rows: list[Row] = []
    ssd_raw: dict = {}
    notes: list[str] = []
    if bytes_per_expert:
        gb_per_token = (n_moe * k * bytes_per_expert) / 1e9
        bw = args.ssd_gbps
        ms_base = gb_per_token / bw * 1000
        ssd_rows.append(Row(["Baseline", f"{gb_per_token*1000:.0f} MB/tok",
                              f"{ms_base:.0f} ms", f"{1000/ms_base:.2f} tok/s", ""]))
        ssd_raw["baseline"] = {"gb_per_token": gb_per_token, "ms": ms_base,
                                "tps": 1000/ms_base}
        for H in WINDOWS:
            hit = window_means.get(H, 0)
            eff_gb = gb_per_token * (1 - hit)
            ms = eff_gb / bw * 1000 if eff_gb > 0 else 1
            tps = 1000 / ms
            cache_mb = H * n_moe * k * bytes_per_expert / 1e6
            ssd_rows.append(Row([
                f"H={H} cache",
                f"{eff_gb*1000:.0f} MB/tok",
                f"{ms:.0f} ms",
                f"{tps:.2f} tok/s",
                f"cache={cache_mb:.0f} MB, hit={hit*100:.0f}%",
            ]))
            ssd_raw[f"h{H}"] = {"gb_per_token": eff_gb, "ms": ms, "tps": tps,
                                  "cache_mb": cache_mb, "hit_rate": hit}
        notes = [
            "NOTE: current Python cache overhead negates these gains.",
            "A native C cache lookup would recover the full saving.",
        ]

    report = BenchReport(
        title="Window cache hit rate — fraction of K experts already in RAM cache",
        subtitle="Cache = union of expert sets from last H decode tokens per layer",
        tables=[
            Table(
                title="Summary",
                headers=["Window", "Mean%", "Median%", "Min%", "Max%", "RAM cost"],
                rows=summary_rows,
            ),
            Table(
                title="Per-layer hit rate (every 4th layer)",
                headers=["Layer", "H=1 hit%", "H=2 hit%", "Unique/tok"],
                rows=layer_rows,
            ),
            Table(
                title=f"Projected SSD savings  (SSD bandwidth = {args.ssd_gbps} GB/s)",
                headers=["Scenario", "MB/tok", "ms/tok", "tok/s", ""],
                rows=ssd_rows,
            ),
        ],
        notes=notes,
        raw={
            "model": args.model,
            "n_tokens": n,
            "n_moe_layers": n_moe,
            "top_k": k,
            "ssd_gbps": args.ssd_gbps,
            "windows": window_raw,
            "per_layer": layer_raw,
            "ssd_savings": ssd_raw,
        },
    )
    report.print_terminal()
    report.save_json(args.output)


if __name__ == "__main__":
    main()
