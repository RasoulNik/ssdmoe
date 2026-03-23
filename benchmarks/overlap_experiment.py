#!/usr/bin/env python3
"""Experiment to measure potential for I/O and compute overlap.

This script measures whether background I/O can complete while GPU
compute is running, which would enable pipelining.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mlx.core as mx
import numpy as np

from streaming_qwen.expert_store import ExpertStore
from streaming_qwen.runtime import build_streamed_model


def measure_io_gpu_overlap(
    expert_store: ExpertStore,
    layer_idx: int,
    expert_indices: list[int],
    hidden_size: int = 2048,
    intermediate_size: int = 512,
    num_trials: int = 10,
) -> dict:
    """Measure if I/O and GPU compute can overlap."""
    layer_info = expert_store.expert_reads[str(layer_idx)]
    components = list(layer_info.keys())

    # Create some dummy GPU work (simulating gather_qmm)
    x = mx.random.normal((1, 1, hidden_size))
    w = mx.random.normal((len(expert_indices), hidden_size, intermediate_size))  # transposed shape

    # Warmup
    for _ in range(3):
        y = x @ w  # simple matmul
        mx.eval(y)

    # Measure I/O only
    io_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = expert_store.read_components_batched(layer_idx, expert_indices, components=components)
        io_times.append(time.perf_counter() - t0)
    avg_io = sum(io_times) / len(io_times)

    # Measure GPU only
    gpu_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        y = x @ w
        mx.eval(y)
        gpu_times.append(time.perf_counter() - t0)
    avg_gpu = sum(gpu_times) / len(gpu_times)

    # Measure sequential: I/O then GPU
    seq_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = expert_store.read_components_batched(layer_idx, expert_indices, components=components)
        y = x @ w
        mx.eval(y)
        seq_times.append(time.perf_counter() - t0)
    avg_seq = sum(seq_times) / len(seq_times)

    # Measure parallel: start I/O in background, then GPU
    par_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()

        # Start I/O in background thread
        io_result = [None]
        io_done = threading.Event()
        def do_io():
            io_result[0] = expert_store.read_components_batched(
                layer_idx, expert_indices, components=components
            )
            io_done.set()

        io_thread = threading.Thread(target=do_io)
        io_thread.start()

        # Do GPU work
        y = x @ w
        mx.eval(y)

        # Wait for I/O
        io_thread.join()

        par_times.append(time.perf_counter() - t0)
    avg_par = sum(par_times) / len(par_times)

    # Calculate overlap efficiency
    # If perfect overlap: parallel time = max(io, gpu)
    # If no overlap: parallel time = io + gpu
    theoretical_best = max(avg_io, avg_gpu)
    theoretical_worst = avg_io + avg_gpu
    overlap_efficiency = (avg_seq - avg_par) / (avg_seq - theoretical_best) if avg_seq > theoretical_best else 0

    return {
        "io_time_ms": avg_io * 1000,
        "gpu_time_ms": avg_gpu * 1000,
        "sequential_time_ms": avg_seq * 1000,
        "parallel_time_ms": avg_par * 1000,
        "theoretical_best_ms": theoretical_best * 1000,
        "overlap_efficiency": overlap_efficiency,
        "speedup": avg_seq / avg_par if avg_par > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--native-reader", default=".run/libexpert_reader.dylib")
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--output", default=".run/overlap-experiment.json")
    args = parser.parse_args()

    # Build model to get expert store
    print("Building model...")
    _, _, expert_store, config = build_streamed_model(
        model_path=Path(args.model),
        index_path=Path(args.index),
        top_k=args.num_experts,
        native_reader_path=Path(args.native_reader) if args.native_reader else None,
    )

    try:
        results = []
        # Test different numbers of experts
        for k in [2, 4, 8]:
            print(f"\nTesting K={k}...")
            # Use first k expert indices
            expert_indices = list(range(k))

            # Test a few layers
            for layer_idx in [0, 20, 39]:
                print(f"  Layer {layer_idx}...")
                result = measure_io_gpu_overlap(
                    expert_store,
                    layer_idx,
                    expert_indices,
                    num_trials=args.num_trials,
                )
                result["k"] = k
                result["layer_idx"] = layer_idx
                results.append(result)
                print(f"    I/O: {result['io_time_ms']:.2f}ms, GPU: {result['gpu_time_ms']:.2f}ms")
                print(f"    Sequential: {result['sequential_time_ms']:.2f}ms, Parallel: {result['parallel_time_ms']:.2f}ms")
                print(f"    Overlap efficiency: {result['overlap_efficiency']:.1%}, Speedup: {result['speedup']:.2f}x")

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {output_path}")

    finally:
        expert_store.close()


if __name__ == "__main__":
    main()
