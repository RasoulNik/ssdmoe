"""Fused expert computation using MLX custom kernels.

Explores fusing multiple gather_qmm + swiglu operations to reduce
kernel launch and intermediate buffer overhead.
"""
from __future__ import annotations

import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu

from .streamed_switch import STREAM_STATS, PROFILE_STREAMED


def fused_gate_up_swiglu(
    x: mx.array,
    gate_w: mx.array,
    gate_s: mx.array,
    gate_b: mx.array,
    up_w: mx.array,
    up_s: mx.array,
    up_b: mx.array,
    indices: mx.array,
    group_size: int = 64,
    bits: int = 4,
    mode: str = "affine",
) -> mx.array:
    """Fused gate projection + up projection + swiglu.

    This combines three operations:
    1. x_gate = gather_qmm(x, gate_w, ...)
    2. x_up = gather_qmm(x, up_w, ...)
    3. activated = swiglu(x_gate, x_up)

    By using @mx.compile, MLX can fuse these into a single kernel.
    """
    x_gate = mx.gather_qmm(
        x,
        gate_w,
        gate_s,
        gate_b,
        rhs_indices=indices,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=False,
    )
    x_up = mx.gather_qmm(
        x,
        up_w,
        up_s,
        up_b,
        rhs_indices=indices,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=False,
    )
    return swiglu(x_gate, x_up)


# Create compiled version
_compiled_fused_gate_up_swiglu = mx.compile(fused_gate_up_swiglu, shapeless=True)


def compute_expert_output_fused(
    x: mx.array,
    tensors: dict[str, mx.array],
    indices: mx.array,
    group_size: int = 64,
    bits: int = 4,
    mode: str = "affine",
    use_compiled: bool = True,
) -> mx.array:
    """Compute expert output with fused gate+up+swiglu.

    This function performs:
    1. Fused: gate_proj + up_proj + swiglu
    2. down_proj

    Using mx.compile for fusion can reduce kernel dispatch overhead.
    """
    x = mx.expand_dims(x, (-2, -3))

    # Fused gate+up+swiglu
    t1 = time.perf_counter()
    if use_compiled:
        activated = _compiled_fused_gate_up_swiglu(
            x,
            tensors["gate_proj.weight"],
            tensors["gate_proj.scales"],
            tensors["gate_proj.biases"],
            tensors["up_proj.weight"],
            tensors["up_proj.scales"],
            tensors["up_proj.biases"],
            indices,
            group_size,
            bits,
            mode,
        )
    else:
        activated = fused_gate_up_swiglu(
            x,
            tensors["gate_proj.weight"],
            tensors["gate_proj.scales"],
            tensors["gate_proj.biases"],
            tensors["up_proj.weight"],
            tensors["up_proj.scales"],
            tensors["up_proj.biases"],
            indices,
            group_size,
            bits,
            mode,
        )
    if PROFILE_STREAMED:
        mx.eval(activated)
    # Track combined time for gate+up+swiglu
    fused_time = time.perf_counter() - t1
    STREAM_STATS["qmm_up_seconds"] += fused_time * 0.4  # Approximate split
    STREAM_STATS["qmm_gate_seconds"] += fused_time * 0.4
    STREAM_STATS["swiglu_seconds"] += fused_time * 0.2

    # Down projection (separate)
    t2 = time.perf_counter()
    x_down = mx.gather_qmm(
        activated,
        tensors["down_proj.weight"],
        tensors["down_proj.scales"],
        tensors["down_proj.biases"],
        rhs_indices=indices,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=False,
    )
    if PROFILE_STREAMED:
        mx.eval(x_down)
    STREAM_STATS["qmm_down_seconds"] += time.perf_counter() - t2

    return x_down.squeeze(-2)


# Microbenchmark function
def benchmark_fused_vs_separate(
    hidden_size: int = 2048,
    intermediate_size: int = 512,
    num_experts: int = 2,
    num_trials: int = 20,
    warmup: int = 5,
) -> dict:
    """Benchmark fused vs separate implementations."""
    # Create test data
    x = mx.random.normal((1, 1, hidden_size))
    x = mx.expand_dims(x, (-2, -3))

    # Create fake quantized weights (using random data)
    gate_w, gate_s, gate_b = mx.quantize(
        mx.random.normal((num_experts, intermediate_size, hidden_size)) * 0.02,
        group_size=64,
        bits=4,
    )
    up_w, up_s, up_b = mx.quantize(
        mx.random.normal((num_experts, intermediate_size, hidden_size)) * 0.02,
        group_size=64,
        bits=4,
    )
    down_w, down_s, down_b = mx.quantize(
        mx.random.normal((num_experts, hidden_size, intermediate_size)) * 0.02,
        group_size=64,
        bits=4,
    )

    indices = mx.arange(num_experts, dtype=mx.int32)

    # Warmup
    for _ in range(warmup):
        x_gate = mx.gather_qmm(
            x, gate_w, gate_s, gate_b,
            rhs_indices=indices, transpose=True, group_size=64, bits=4, sorted_indices=False,
        )
        x_up = mx.gather_qmm(
            x, up_w, up_s, up_b,
            rhs_indices=indices, transpose=True, group_size=64, bits=4, sorted_indices=False,
        )
        activated = swiglu(x_gate, x_up)
        x_down = mx.gather_qmm(
            activated, down_w, down_s, down_b,
            rhs_indices=indices, transpose=True, group_size=64, bits=4, sorted_indices=False,
        )
        mx.eval(x_down)

    # Benchmark separate operations
    separate_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        x_gate = mx.gather_qmm(
            x, gate_w, gate_s, gate_b,
            rhs_indices=indices, transpose=True, group_size=64, bits=4, sorted_indices=False,
        )
        x_up = mx.gather_qmm(
            x, up_w, up_s, up_b,
            rhs_indices=indices, transpose=True, group_size=64, bits=4, sorted_indices=False,
        )
        activated = swiglu(x_gate, x_up)
        x_down = mx.gather_qmm(
            activated, down_w, down_s, down_b,
            rhs_indices=indices, transpose=True, group_size=64, bits=4, sorted_indices=False,
        )
        mx.eval(x_down)
        separate_times.append((time.perf_counter() - t0) * 1000)

    # Benchmark compiled fused version
    fused_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        activated = _compiled_fused_gate_up_swiglu(
            x, gate_w, gate_s, gate_b, up_w, up_s, up_b, indices, 64, 4, "affine",
        )
        x_down = mx.gather_qmm(
            activated, down_w, down_s, down_b,
            rhs_indices=indices, transpose=True, group_size=64, bits=4, sorted_indices=False,
        )
        mx.eval(x_down)
        fused_times.append((time.perf_counter() - t0) * 1000)

    return {
        "separate_ms": sum(separate_times) / len(separate_times),
        "fused_ms": sum(fused_times) / len(fused_times),
        "speedup": sum(separate_times) / sum(fused_times) if sum(fused_times) > 0 else 0,
        "num_experts": num_experts,
    }


if __name__ == "__main__":
    print("Benchmarking fused vs separate expert computation...")
    for k in [2, 4, 8]:
        result = benchmark_fused_vs_separate(num_experts=k, num_trials=50)
        print(f"K={k}: separate={result['separate_ms']:.3f}ms, fused={result['fused_ms']:.3f}ms, speedup={result['speedup']:.2f}x")
