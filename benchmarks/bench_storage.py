#!/usr/bin/env python3
"""SSD / storage-layer microbenchmarks for routed expert reads.

Two modes (select with --mode):
  expert     — read individual experts via pread() (ExpertStore.read_experts_parallel)
  component  — read all components for selected experts via native batch reader

Mode 'expert' tests the Python-ThreadPool parallel-pread path.
Mode 'component' tests the native C reader batched path and requires --native-reader.

These benchmarks are model-agnostic — they only need an expert_index.json file
and do not load or execute the model.

Examples:
  # Expert-level pread benchmark (layer 0, 8 experts, 10 iterations)
  poetry run python benchmarks/bench_storage.py --mode expert \\
    --index .run/qwen35b-expert-index.json --layer 0

  # Component-level native batch benchmark (all layers, 5 sweeps)
  poetry run python benchmarks/bench_storage.py --mode component \\
    --index .run/nemotron30b-expert-index.json \\
    --native-reader .run/libexpert_reader.dylib --layers all

  # Both modes in sequence
  poetry run python benchmarks/bench_storage.py --mode both \\
    --index .run/qwen35b-expert-index.json \\
    --native-reader .run/libexpert_reader.dylib
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lib.loader import save_json  # noqa: E402

from streaming_qwen.expert_store import ExpertStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSD microbenchmarks for routed expert reads")
    p.add_argument("--mode", choices=["expert", "component", "both"], default="component",
                   help="expert=pread per expert, component=native batch reader, both=run both")
    p.add_argument("--index", required=True, help="Path to expert_index.json")
    p.add_argument("--experts", default="0,1,2,3,4,5,6,7",
                   help="Comma-separated expert ids to read")
    # expert mode
    p.add_argument("--layer", type=int, default=0, help="[expert mode] Layer index to benchmark")
    p.add_argument("--workers", type=int, default=None,
                   help="[expert mode] Parallel worker count (default: number of experts)")
    # component mode
    p.add_argument("--layers", default="all",
                   help="[component mode] Comma-separated layer ids or 'all'")
    p.add_argument("--native-reader", default=None, help="[component mode] Path to native reader dylib")
    p.add_argument("--component-workers", type=int, default=3,
                   help="[component mode] Parallel component workers")
    p.add_argument("--iters", type=int, default=10, help="Number of iterations / sweeps")
    p.add_argument("--output", default=None, help="Optional JSON output path")
    return p.parse_args()


def _parse_layers(raw: str, index: dict) -> list[int]:
    if raw == "all":
        return sorted(int(k) for k in index["expert_reads"].keys())
    return [int(x) for x in raw.split(",") if x.strip()]


def run_expert_mode(args, expert_ids: list[int]) -> dict:
    """Benchmark read_experts_parallel on a single layer."""
    samples: list[float] = []
    total_bytes = 0

    with ExpertStore(Path(args.index)) as store:
        for _ in range(args.iters):
            t0 = time.perf_counter()
            payload = store.read_experts_parallel(args.layer, expert_ids, max_workers=args.workers)
            samples.append(time.perf_counter() - t0)
            if not total_bytes:
                total_bytes = sum(
                    len(b) for expert in payload.values() for b in expert.values()
                )

    gib = total_bytes / 1024**3
    mean_s = statistics.mean(samples)
    p50_s = statistics.median(samples)
    best_s = min(samples)

    result = {
        "mode": "expert",
        "layer": args.layer,
        "experts": expert_ids,
        "iters": args.iters,
        "payload_gib": round(gib, 4),
        "mean_ms": round(mean_s * 1000, 2),
        "p50_ms": round(p50_s * 1000, 2),
        "best_ms": round(best_s * 1000, 2),
        "mean_gibs": round(gib / mean_s, 3),
        "p50_gibs": round(gib / p50_s, 3),
        "best_gibs": round(gib / best_s, 3),
    }
    print(f"\n[expert mode]  layer={args.layer}  experts={expert_ids}  payload={gib:.4f} GiB")
    print(f"  Mean:  {result['mean_ms']:>8.2f} ms  ({result['mean_gibs']:.3f} GiB/s)")
    print(f"  P50:   {result['p50_ms']:>8.2f} ms  ({result['p50_gibs']:.3f} GiB/s)")
    print(f"  Best:  {result['best_ms']:>8.2f} ms  ({result['best_gibs']:.3f} GiB/s)")
    return result


def run_component_mode(args, expert_ids: list[int]) -> dict:
    """Benchmark read_components_batched across layers using native reader."""
    if not args.native_reader:
        raise ValueError("--native-reader is required for component mode")

    with Path(args.index).open() as f:
        index = json.load(f)
    layers = _parse_layers(args.layers, index)

    layer_samples: dict[int, list[float]] = {layer: [] for layer in layers}
    sweep_samples: list[float] = []
    bytes_per_layer = None

    with ExpertStore(
        Path(args.index),
        native_reader_path=Path(args.native_reader),
        component_workers=args.component_workers,
    ) as store:
        for _ in range(args.iters):
            sweep_start = time.perf_counter()
            for layer in layers:
                t0 = time.perf_counter()
                payload = store.read_components_batched(layer, expert_ids)
                layer_samples[layer].append(time.perf_counter() - t0)
                if bytes_per_layer is None:
                    bytes_per_layer = sum(len(blob) for blob in payload.values())
            sweep_samples.append(time.perf_counter() - sweep_start)

        store_stats = dict(store.stats)

    if bytes_per_layer is None:
        raise RuntimeError("No payload was read")

    all_layer_s = [s for samples in layer_samples.values() for s in samples]
    total_bytes = bytes_per_layer * len(layers) * args.iters

    result = {
        "mode": "component",
        "experts": expert_ids,
        "layers": layers,
        "iters": args.iters,
        "component_workers": args.component_workers,
        "bytes_per_layer": bytes_per_layer,
        "payload_gib_per_layer": round(bytes_per_layer / 1024**3, 4),
        "mean_layer_ms": round(statistics.mean(all_layer_s) * 1000, 2),
        "p50_layer_ms": round(statistics.median(all_layer_s) * 1000, 2),
        "best_layer_ms": round(min(all_layer_s) * 1000, 2),
        "mean_sweep_ms": round(statistics.mean(sweep_samples) * 1000, 2),
        "aggregate_gibs": round((total_bytes / 1024**3) / sum(sweep_samples), 3),
        "expert_store": store_stats,
    }
    print(f"\n[component mode]  layers={layers[0]}..{layers[-1]} ({len(layers)})  "
          f"experts={expert_ids}  payload_per_layer={result['payload_gib_per_layer']:.4f} GiB")
    print(f"  Layer mean/p50/best: {result['mean_layer_ms']:.2f} / "
          f"{result['p50_layer_ms']:.2f} / {result['best_layer_ms']:.2f} ms")
    print(f"  Sweep mean: {result['mean_sweep_ms']:.2f} ms")
    print(f"  Aggregate throughput: {result['aggregate_gibs']:.3f} GiB/s")
    return result


def main() -> None:
    args = parse_args()
    expert_ids = [int(x) for x in args.experts.split(",") if x.strip()]
    results = {}

    if args.mode in ("expert", "both"):
        results["expert"] = run_expert_mode(args, expert_ids)

    if args.mode in ("component", "both"):
        results["component"] = run_component_mode(args, expert_ids)

    save_json(results, args.output)


if __name__ == "__main__":
    main()
