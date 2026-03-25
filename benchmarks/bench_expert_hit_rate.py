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
    p.add_argument("--model", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--native-reader", required=True)
    p.add_argument("--tokens", type=int, default=64, help="Decode tokens to observe")
    p.add_argument("--routed-top-k", type=int, default=8)
    p.add_argument("--component-workers", type=int, default=3)
    p.add_argument("--prompt", default="Explain how neural networks work in detail.")
    return p.parse_args()


# ─── Recording hook ────────────────────────────────────────────────────────────

# selections[layer_idx] = list of frozenset per token (decode order)
_selections: dict[int, list[frozenset]] = defaultdict(list)
_token_counter = 0


class _RecordingSwitch(nn.Module):
    """Thin wrapper that records expert selections then delegates to the real switch."""

    def __init__(self, inner, layer_idx: int):
        super().__init__()
        self._inner = inner
        self._layer_idx = layer_idx

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        indices_np = np.array(indices.tolist(), dtype=np.int32)
        selected = frozenset(np.unique(indices_np).tolist())
        _selections[self._layer_idx].append(selected)
        # Re-create indices as mx.array so the inner switch can call tolist() again
        return self._inner(x, indices)


def install_recording_hooks(model) -> list[int]:
    """Replace all StreamedSwitchGLU layers with _RecordingSwitch wrappers.
    Returns list of layer indices that were patched."""
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
            moe_module.switch_mlp = _RecordingSwitch(switch, i)
            patched.append(i)
    return patched


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
    patched_layers = install_recording_hooks(model)
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

    # ── Compute hit rates ──
    # Per-layer hit rate: fraction of consecutive token pairs that selected the same experts
    layer_hit_rates = {}
    for layer_idx in patched_layers:
        sel = _selections[layer_idx]
        if len(sel) < 2:
            continue
        hits = sum(1 for a, b in zip(sel, sel[1:]) if a == b)
        layer_hit_rates[layer_idx] = hits / (len(sel) - 1)

    if not layer_hit_rates:
        print("No data collected.")
        return

    rates = list(layer_hit_rates.values())
    mean_rate = np.mean(rates)
    median_rate = np.median(rates)
    min_rate = np.min(rates)
    max_rate = np.max(rates)

    print("═" * 60)
    print("Cross-token expert selection hit rate (same experts → cache hit)")
    print("─" * 60)
    print(f"  Mean:    {mean_rate*100:5.1f}%")
    print(f"  Median:  {median_rate*100:5.1f}%")
    print(f"  Min:     {min_rate*100:5.1f}%")
    print(f"  Max:     {max_rate*100:5.1f}%")
    print()

    # Per-layer table (show every 5th layer to keep output concise)
    print("Per-layer hit rate (sample):")
    print(f"  {'Layer':>6}  {'Tokens':>6}  {'Hit%':>6}  {'Unique sets':>11}")
    print("  " + "-" * 34)
    for layer_idx in patched_layers[::4]:
        sel = _selections[layer_idx]
        hr = layer_hit_rates.get(layer_idx, 0)
        n_unique = len(set(sel))
        print(f"  {layer_idx:>6}  {len(sel):>6}  {hr*100:>5.1f}%  {n_unique:>11}")
    print()

    # ── Jaccard similarity (partial overlap) ──
    # Even when the set isn't identical, how much do they overlap?
    all_jaccard = []
    for layer_idx in patched_layers:
        sel = _selections[layer_idx]
        for a, b in zip(sel, sel[1:]):
            if not a and not b:
                continue
            jaccard = len(a & b) / len(a | b) if (a | b) else 1.0
            all_jaccard.append(jaccard)
    if all_jaccard:
        print(f"Jaccard similarity between consecutive tokens (partial overlap):")
        print(f"  Mean:    {np.mean(all_jaccard)*100:5.1f}%")
        print(f"  Median:  {np.median(all_jaccard)*100:5.1f}%")
        print(f"  Min:     {np.min(all_jaccard)*100:5.1f}%")
        print()

    # ── SSD read volume math ──
    # Get expert size from expert_store
    try:
        n_moe_layers = len(patched_layers)
        k = args.routed_top_k
        # Sample one layer's component info to estimate expert size
        sample_layer = str(patched_layers[0])
        layer_info = expert_store.expert_reads[sample_layer]
        # expert_size is the per-expert byte count; sum across all streamed components
        bytes_per_expert = sum(info.get("expert_size", 0) for info in layer_info.values())
        mb_per_expert = bytes_per_expert / 1e6
        mib_per_token = (n_moe_layers * k * bytes_per_expert) / 2**20
        gb_per_token = (n_moe_layers * k * bytes_per_expert) / 1e9

        print("═" * 60)
        print("SSD read volume per decode token")
        print("─" * 60)
        print(f"  MoE layers:          {n_moe_layers}")
        print(f"  Top-K experts:       {k}")
        print(f"  Expert size:         {mb_per_expert:.2f} MB ({bytes_per_expert/1024:.0f} KB)")
        print(f"  Total per token:     {gb_per_token*1000:.0f} MB  ({mib_per_token:.0f} MiB)")
        print()

        # At various SSD bandwidths
        for bw_gbps, label in [(3.0, "slow SSD"), (3.4, "typical M4"), (6.0, "fast NVMe")]:
            ms = gb_per_token / bw_gbps * 1000
            tps = 1000 / ms
            print(f"  @ {bw_gbps} GB/s ({label:12s}): {ms:.0f} ms/token  ({tps:.2f} tok/s SSD-limited)")

        print()
        eff_hit = mean_rate  # fraction of (layer, token) pairs that are cache hits
        read_reduction = eff_hit  # if hit, no pread needed
        effective_gb = gb_per_token * (1 - read_reduction)
        if effective_gb > 0:
            ms_with_cache = effective_gb / 3.4 * 1000
            print(f"  With {mean_rate*100:.0f}% hit rate + perfect cross-token cache:")
            print(f"    Effective read:  {effective_gb*1000:.0f} MB/token")
            print(f"    SSD time:        {ms_with_cache:.0f} ms/token")
        else:
            print(f"  With {mean_rate*100:.0f}% hit rate: all reads eliminated!")
    except Exception as e:
        print(f"  (could not compute SSD math: {e})")

    print()
    print("═" * 60)
    print("Interpretation")
    print("─" * 60)
    if mean_rate > 0.7:
        print(f"  Hit rate {mean_rate*100:.0f}% is HIGH → cross-token expert cache would help.")
        print(f"  GIL-free native prefetch could potentially eliminate most SSD latency.")
    elif mean_rate > 0.3:
        print(f"  Hit rate {mean_rate*100:.0f}% is MODERATE → cache helps but SSD reads remain.")
    else:
        print(f"  Hit rate {mean_rate*100:.0f}% is LOW → experts change frequently, cache barely helps.")
        print(f"  SSD reads are unavoidable at current routing diversity.")


if __name__ == "__main__":
    main()
