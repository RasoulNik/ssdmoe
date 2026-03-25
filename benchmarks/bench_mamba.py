#!/usr/bin/env python3
"""Benchmark individual Mamba layers in isolation.

Measures:
  - Wall time per Mamba layer (GPU compute, forced sync)
  - Effective memory bandwidth (weight bytes read / time)
  - Breakdown: in_proj / conv / ssm_kernel / norm / out_proj

Usage:
  poetry run python benchmarks/bench_mamba.py \\
    --model ~/.cache/huggingface/hub/models--sjug--Nemotron-3-Super-120B-A12B-MLX-4bit/snapshots/ff505b4c07e1c23d8e650e9e37877bdf71c9424b
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--index", required=True, help="Expert index JSON")
    p.add_argument("--native-reader", default=None, help="Path to native reader dylib")
    p.add_argument("--warmup", type=int, default=5, help="Warmup passes per layer")
    p.add_argument("--reps", type=int, default=20, help="Measured passes per layer")
    p.add_argument("--layers", type=int, default=None, help="How many Mamba layers to bench (default: all)")
    return p.parse_args()


def _weight_bytes(module) -> int:
    """Estimate bytes of weight data read from DRAM per forward pass."""
    total = 0
    for _, w in module.parameters().items() if hasattr(module.parameters(), 'items') else []:
        if isinstance(w, mx.array):
            total += w.nbytes
    # Flatten nested params
    def _sum(params):
        if isinstance(params, dict):
            return sum(_sum(v) for v in params.values())
        if isinstance(params, list):
            return sum(_sum(v) for v in params)
        if isinstance(params, mx.array):
            return params.nbytes
        return 0
    return _sum(module.parameters())


def bench_layer(layer, x, cache_entry, warmup, reps):
    """Run a single Mamba layer repeatedly; return (mean_ms, std_ms)."""
    times = []

    for i in range(warmup + reps):
        # Create a fresh cache each call so state doesn't accumulate
        c = ArraysCache(size=2)

        t0 = time.perf_counter()
        out = layer(x, mask=None, cache=c)
        mx.eval(out)
        t1 = time.perf_counter()

        if i >= warmup:
            times.append((t1 - t0) * 1000)

    return float(np.mean(times)), float(np.std(times))


def bench_sub_ops(mixer, x_norm, warmup, reps):
    """Break down time inside NemotronHMamba2Mixer into sub-operations."""
    def _time(fn, w=warmup, r=reps):
        ts = []
        for i in range(w + r):
            t0 = time.perf_counter()
            out = fn()
            mx.eval(out)
            t1 = time.perf_counter()
            if i >= w:
                ts.append((t1 - t0) * 1000)
        return float(np.mean(ts)), float(np.std(ts))

    # in_proj
    t_inproj, s_inproj = _time(lambda: mixer.in_proj(x_norm))

    # Full forward through in_proj to get inputs for subsequent steps
    projected = mixer.in_proj(x_norm)
    mx.eval(projected)
    gate, conv_input, dt = mx.split(
        projected,
        [mixer.intermediate_size, mixer.intermediate_size + mixer.conv_dim],
        axis=-1,
    )
    mx.eval(gate, conv_input, dt)

    # conv
    t_conv, s_conv = _time(lambda: mixer._conv(conv_input, None, None))

    conv_out = mixer._conv(conv_input, None, None)
    mx.eval(conv_out)
    hidden_states_ssm, B, C_ssm = mx.split(
        conv_out,
        [mixer.intermediate_size, mixer.intermediate_size + mixer.n_groups * mixer.ssm_state_size],
        axis=-1,
    )
    hidden_states_r = hidden_states_ssm.reshape(
        hidden_states_ssm.shape[0], hidden_states_ssm.shape[1],
        mixer.num_heads, mixer.head_dim,
    )
    B_r = B.reshape(B.shape[0], B.shape[1], mixer.n_groups, mixer.ssm_state_size)
    C_r = C_ssm.reshape(C_ssm.shape[0], C_ssm.shape[1], mixer.n_groups, mixer.ssm_state_size)

    # Build a dummy SSM state
    from mlx_lm.models.ssm import compute_dt
    dummy_state = mx.zeros(
        (1, mixer.num_heads, mixer.head_dim, mixer.ssm_state_size),
        dtype=x_norm.dtype,
    )
    mx.eval(dummy_state)

    dt_computed = compute_dt(dt, mixer.dt_bias, mixer.time_step_limit)
    mx.eval(dt_computed)

    # ssm kernel (using the fast Metal path directly)
    from mlx_lm.models.ssm import ssm_update_kernel
    t_ssm, s_ssm = _time(lambda: ssm_update_kernel(
        hidden_states_r, mixer.A_log, B_r, C_r,
        mixer.D.astype(x_norm.dtype), dt_computed, mixer.dt_bias,
        dummy_state, mixer.time_step_limit,
    )[0])

    # norm
    y_ssm, _ = ssm_update_kernel(
        hidden_states_r, mixer.A_log, B_r, C_r,
        mixer.D.astype(x_norm.dtype), dt_computed, mixer.dt_bias,
        dummy_state, mixer.time_step_limit,
    )
    y_ssm = y_ssm.reshape(1, 1, mixer.intermediate_size)
    mx.eval(y_ssm)
    t_norm, s_norm = _time(lambda: mixer.norm(y_ssm, gate))

    # out_proj
    y_normed = mixer.norm(y_ssm, gate)
    mx.eval(y_normed)
    t_outproj, s_outproj = _time(lambda: mixer.out_proj(y_normed))

    return {
        "in_proj":  (t_inproj,  s_inproj),
        "conv1d":   (t_conv,    s_conv),
        "ssm_kernel": (t_ssm,  s_ssm),
        "norm+gate": (t_norm,  s_norm),
        "out_proj": (t_outproj, s_outproj),
    }


def main():
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()

    index_path = Path(args.index).expanduser().resolve()

    print(f"Loading model from {model_path}")
    from streaming_moe.runtime import build_streamed_model
    model, tokenizer, expert_store, config = build_streamed_model(
        model_path, index_path,
        native_reader_path=Path(args.native_reader) if args.native_reader else None,
    )
    print("Model loaded.\n")

    # Collect Mamba layers
    text_model = getattr(getattr(model, "language_model", model), "model", model)
    backbone = getattr(text_model, "backbone", text_model)
    all_layers = list(backbone.layers)
    mamba_layers = [(i, l) for i, l in enumerate(all_layers) if l.block_type == "M"]
    print(f"Total layers: {len(all_layers)}  Mamba layers: {len(mamba_layers)}")

    if args.layers is not None:
        mamba_layers = mamba_layers[:args.layers]
    print(f"Benchmarking {len(mamba_layers)} Mamba layer(s).\n")

    hidden_size = config.get("hidden_size", 4096)
    x = mx.random.normal((1, 1, hidden_size)).astype(mx.bfloat16)
    mx.eval(x)

    # --- Per-layer timing ---
    print(f"{'Layer':>6}  {'mean ms':>8}  {'std ms':>7}  {'GB/s (weights)':>15}  {'weight MB':>10}")
    print("-" * 60)

    all_times = []
    for layer_pos, (global_idx, layer) in enumerate(mamba_layers):
        mean_ms, std_ms = bench_layer(layer, x, None, args.warmup, args.reps)
        w_bytes = _weight_bytes(layer)
        w_mb = w_bytes / 1e6
        gbps = (w_bytes / 1e9) / (mean_ms / 1000) if mean_ms > 0 else 0
        all_times.append(mean_ms)
        print(f"{global_idx:>6}  {mean_ms:>8.3f}  {std_ms:>7.3f}  {gbps:>15.2f}  {w_mb:>10.1f}")

    print("-" * 60)
    total_ms = sum(all_times)
    mean_all = float(np.mean(all_times))
    print(f"\nAll {len(mamba_layers)} Mamba layers total: {total_ms:.1f} ms/token")
    print(f"Mean per layer: {mean_all:.3f} ms")

    # --- Sub-operation breakdown on first Mamba layer ---
    print(f"\n=== Sub-op breakdown (layer {mamba_layers[0][0]}) ===")
    first_layer = mamba_layers[0][1]
    mixer = first_layer.mixer

    # Normalize input (as the layer itself would)
    x_norm = first_layer.norm(x)
    mx.eval(x_norm)

    breakdown = bench_sub_ops(mixer, x_norm, args.warmup, args.reps)
    total_sub = sum(v[0] for v in breakdown.values())
    print(f"{'Op':>12}  {'ms':>7}  {'%':>6}")
    print("-" * 30)
    for op, (mean_ms, std_ms) in breakdown.items():
        pct = 100 * mean_ms / total_sub if total_sub > 0 else 0
        print(f"{op:>12}  {mean_ms:>7.3f}  {pct:>5.1f}%")
    print("-" * 30)
    print(f"{'sum':>12}  {total_sub:>7.3f}")

    # --- Weight sizes ---
    print(f"\n=== Weight sizes (layer {mamba_layers[0][0]}) ===")
    def _print_weights(module, prefix=""):
        for name, child in module.named_modules() if hasattr(module, 'named_modules') else []:
            pass
    # Manual inspection
    m = first_layer.mixer
    for attr in ["in_proj", "out_proj", "conv1d", "norm"]:
        mod = getattr(m, attr, None)
        if mod is not None:
            b = _weight_bytes(mod)
            print(f"  {attr:12s}: {b/1e6:.1f} MB")
    print(f"  {'A_log/D/dt_bias':12s}: {(m.A_log.nbytes + m.D.nbytes + m.dt_bias.nbytes)/1e3:.1f} KB")


if __name__ == "__main__":
    main()
