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
    p.add_argument("--model", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--native-reader", required=True)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--routed-top-k", type=int, default=8)
    p.add_argument("--component-workers", type=int, default=3)
    p.add_argument("--prompt", default="Explain the architecture of transformer models.")
    p.add_argument("--output", default=None)
    return p.parse_args()


# ── Recording hooks ───────────────────────────────────────────────────────────

# Per-layer: list of (router_input_hidden, frozenset_of_selected_experts)
_layer_data: dict[int, list[tuple[np.ndarray, frozenset]]] = defaultdict(list)


class _RecordingMoEWrapper(nn.Module):
    """Wraps the entire MoE module to capture hidden state + expert selections."""

    def __init__(self, inner_moe, layer_idx: int):
        super().__init__()
        self._inner = inner_moe
        self._layer_idx = layer_idx

    def __call__(self, x: mx.array) -> mx.array:
        # x is the router input (hidden state entering this MoE layer)
        # Record it before the forward pass
        x_np = np.array(x.tolist(), dtype=np.float32).reshape(-1)[:64]  # first 64 dims as proxy
        result = self._inner(x)
        # The inner call triggers expert selection — but we need the indices
        # We'll get them from the switch hook below
        _layer_data[self._layer_idx + 1000].append(x_np)  # store with offset key
        return result


class _RecordingSwitch(nn.Module):
    """Records expert selections at each layer."""

    def __init__(self, inner, layer_idx: int):
        super().__init__()
        self._inner = inner
        self._layer_idx = layer_idx

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        indices_np = np.array(indices.tolist(), dtype=np.int32)
        selected = frozenset(np.unique(indices_np).tolist())
        _layer_data[self._layer_idx].append(selected)
        return self._inner(x, indices)


def install_hooks(model) -> list[int]:
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


def run_decode(model, tokenizer, prompt: str, n_tokens: int, top_k: int) -> None:
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
    n = 0
    for token, _ in generate_step(last, model, max_tokens=n_tokens, prompt_cache=cache):
        mx.eval(token)
        n += 1
        if int(token) in eos_ids:
            break


def main():
    args = parse_args()

    print("Loading model …")
    model, tokenizer, expert_store, _ = build_streamed_model(
        Path(args.model), Path(args.index),
        native_reader_path=Path(args.native_reader),
        component_workers=args.component_workers,
    )
    patched = install_hooks(model)
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

    print(f"\n  Prediction = 'same experts as previous layer'")
    print(f"  Hit rate (fraction of L+1 experts that appeared in L):")
    print(f"    Mean:   {np.mean(all_hits)*100:5.1f}%")
    print(f"    Median: {np.median(all_hits)*100:5.1f}%")
    print(f"    Min:    {np.min(all_hits)*100:5.1f}%")
    print(f"    Max:    {np.max(all_hits)*100:5.1f}%")
    print()
    print(f"  Jaccard (set overlap L vs L+1):")
    print(f"    Mean:   {np.mean(all_jaccards)*100:5.1f}%")
    print(f"    Median: {np.median(all_jaccards)*100:5.1f}%")
    print()

    # Per-pair table (every 4th)
    print("  Per adjacent-layer pair (every 4th):")
    print(f"  {'L':>4}→{'L+1':>4}  {'hit%':>7}  {'jaccard%':>9}")
    print("  " + "─" * 28)
    for l_cur, l_next, mh, mj, n in per_pair[::4]:
        print(f"  {l_cur:>4}→{l_next:>4}  {mh*100:>6.1f}%  {mj*100:>8.1f}%")

    # ── SSD savings projection ──
    sample_layer = str(patched[0])
    layer_info = expert_store.expert_reads[sample_layer]
    bytes_per_expert = sum(info.get("expert_size", 0) for info in layer_info.values())
    n_moe = len(patched)
    k = args.routed_top_k
    gb_per_token = (n_moe * k * bytes_per_expert) / 1e9
    bw = 3.4
    mean_hit = float(np.mean(all_hits))

    print()
    print("═" * 65)
    print("SSD prefetch savings with intra-model next-layer prediction")
    print("─" * 65)
    ms_base = gb_per_token / bw * 1000
    eff_gb   = gb_per_token * (1 - mean_hit)
    ms_pred  = eff_gb / bw * 1000
    print(f"  Baseline:         {gb_per_token*1000:.0f} MB  {ms_base:.0f} ms  {1000/ms_base:.2f} tok/s")
    print(f"  With {mean_hit*100:.0f}% hit rate:  {eff_gb*1000:.0f} MB  {ms_pred:.0f} ms  {1000/ms_pred:.2f} tok/s  (theoretical max)")
    print()
    print("  NOTE: 'same experts as previous layer' is the WEAKEST predictor.")
    print("  Quasi-hidden-state predictor (Speculating Experts) achieves 84-91%.")
    print("  Pre-attention linear predictor (2511.10676) achieves 93-97%.")

    # ── Save ──
    results = {
        "model": args.model,
        "tokens": args.tokens,
        "top_k": args.routed_top_k,
        "predictor": "same_as_prev_layer",
        "hit_rate": {"mean": float(np.mean(all_hits)), "median": float(np.median(all_hits))},
        "jaccard":  {"mean": float(np.mean(all_jaccards)), "median": float(np.median(all_jaccards))},
        "per_pair": [
            {"layer": lc, "next_layer": ln, "hit_rate": mh, "jaccard": mj, "n_tokens": n}
            for lc, ln, mh, mj, n in per_pair
        ],
    }
    out = args.output or "benchmarks/results/next-layer-prediction.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
