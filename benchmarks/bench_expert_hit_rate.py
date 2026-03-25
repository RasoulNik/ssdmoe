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

    # ── Summary table ──
    print("═" * 68)
    print("Window cache hit rate  — fraction of K experts already in RAM cache")
    print("  Cache = union of expert sets from last H decode tokens per layer")
    print("─" * 68)
    print(f"  {'Window':>8}  {'Mean%':>7}  {'Median%':>8}  {'Min%':>6}  {'Max%':>6}  {'RAM cost':>10}")
    print("  " + "─" * 54)

    window_means: dict[int, float] = {}
    try:
        sample_layer = str(patched_layers[0])
        layer_info = expert_store.expert_reads[sample_layer]
        bytes_per_expert = sum(info.get("expert_size", 0) for info in layer_info.values())
        n_moe = len(patched_layers)
        k = args.routed_top_k
    except Exception:
        bytes_per_expert = 0; n_moe = 0; k = 0

    for H in WINDOWS:
        fracs = window_hits[H]
        if not fracs:
            continue
        mean_h   = float(np.mean(fracs))
        med_h    = float(np.median(fracs))
        min_h    = float(np.min(fracs))
        max_h    = float(np.max(fracs))
        window_means[H] = mean_h
        cache_mb = H * n_moe * k * bytes_per_expert / 1e6 if bytes_per_expert else 0
        print(f"  H={H} ({H} tok{'s' if H>1 else ' '}):  {mean_h*100:>6.1f}%  {med_h*100:>7.1f}%  "
              f"{min_h*100:>5.1f}%  {max_h*100:>5.1f}%  {cache_mb:>7.0f} MB")
    print()

    # ── Per-layer breakdown ──
    print("Per-layer hit rate (every 4th layer):")
    print(f"  {'Layer':>6}  {'H=1 hit%':>9}  {'H=2 hit%':>9}  {'Unique/tok':>10}")
    print("  " + "─" * 42)
    for layer_idx in patched_layers[::4]:
        sel = _selections[layer_idx]
        h1 = window_hits_per_layer[1].get(layer_idx, [])
        h2 = window_hits_per_layer[2].get(layer_idx, [])
        n_unique = len(set(sel))
        r1 = f"{np.mean(h1)*100:>8.1f}%" if h1 else f"{'—':>8} "
        r2 = f"{np.mean(h2)*100:>8.1f}%" if h2 else f"{'—':>8} "
        print(f"  {layer_idx:>6}  {r1}  {r2}  {n_unique:>5}/{len(sel)}")
    print()

    # ── SSD savings projection ──
    if bytes_per_expert:
        gb_per_token = (n_moe * k * bytes_per_expert) / 1e9
        bw = 3.4
        print("═" * 68)
        print("Projected SSD savings with a native (GIL-free) window cache")
        print("─" * 68)
        ms_base = gb_per_token / bw * 1000
        print(f"  Baseline:      {gb_per_token*1000:>6.0f} MB/tok  {ms_base:>6.0f} ms  {1000/ms_base:>5.2f} tok/s")
        for H in WINDOWS:
            hit = window_means.get(H, 0)
            eff_gb = gb_per_token * (1 - hit)
            ms = eff_gb / bw * 1000 if eff_gb > 0 else 1
            tps = 1000 / ms
            cache_mb = H * n_moe * k * bytes_per_expert / 1e6
            print(f"  H={H} cache:    {eff_gb*1000:>6.0f} MB/tok  {ms:>6.0f} ms  {tps:>5.2f} tok/s  "
                  f"(cache={cache_mb:.0f} MB RAM, hit={hit*100:.0f}%)")
        print()
        print("  NOTE: current Python cache overhead negates these gains.")
        print("  A native C cache lookup would recover the full saving.")


if __name__ == "__main__":
    main()
