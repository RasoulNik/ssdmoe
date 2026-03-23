#!/usr/bin/env python3
"""Benchmark layer-level expert component loading under different parallelism modes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from streaming_qwen.expert_store import ExpertStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark layer-level expert loading")
    parser.add_argument("--index", required=True, help="Path to expert_index.json")
    parser.add_argument(
        "--experts",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated expert ids to read per layer",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated layer ids, or 'all'",
    )
    parser.add_argument("--iters", type=int, default=5, help="Number of sweeps")
    parser.add_argument(
        "--component-workers",
        type=int,
        default=3,
        help="Parallel component workers in ExpertStore",
    )
    parser.add_argument(
        "--native-reader",
        required=True,
        help="Path to native reader dylib",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path",
    )
    return parser.parse_args()


def parse_layers(raw: str, index: dict) -> list[int]:
    if raw == "all":
        return sorted(int(k) for k in index["expert_reads"].keys())
    return [int(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    expert_ids = [int(x) for x in args.experts.split(",") if x.strip()]

    with Path(args.index).open() as f:
        index = json.load(f)
    layers = parse_layers(args.layers, index)

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
                elapsed = time.perf_counter() - t0
                layer_samples[layer].append(elapsed)
                if bytes_per_layer is None:
                    bytes_per_layer = sum(len(blob) for blob in payload.values())
            sweep_samples.append(time.perf_counter() - sweep_start)

        stats = dict(store.stats)

    if bytes_per_layer is None:
        raise RuntimeError("No payload was read")

    total_bytes = bytes_per_layer * len(layers) * args.iters
    result = {
        "experts": expert_ids,
        "layers": layers,
        "iters": args.iters,
        "component_workers": args.component_workers,
        "native_reader": str(Path(args.native_reader).resolve()),
        "bytes_per_layer": bytes_per_layer,
        "payload_gib_per_layer": bytes_per_layer / 1024**3,
        "mean_layer_ms": statistics.mean(
            elapsed for samples in layer_samples.values() for elapsed in samples
        )
        * 1000.0,
        "p50_layer_ms": statistics.median(
            elapsed for samples in layer_samples.values() for elapsed in samples
        )
        * 1000.0,
        "best_layer_ms": min(elapsed for samples in layer_samples.values() for elapsed in samples)
        * 1000.0,
        "mean_sweep_ms": statistics.mean(sweep_samples) * 1000.0,
        "aggregate_gib_per_s": (total_bytes / 1024**3) / sum(sweep_samples),
        "expert_store": stats,
    }

    print(f"Experts: {expert_ids}")
    print(f"Layers: {layers[0]}..{layers[-1]} ({len(layers)} total)")
    print(f"Payload per layer: {result['payload_gib_per_layer']:.4f} GiB")
    print(
        f"Layer mean/p50/best: {result['mean_layer_ms']:.2f} / "
        f"{result['p50_layer_ms']:.2f} / {result['best_layer_ms']:.2f} ms"
    )
    print(f"Sweep mean: {result['mean_sweep_ms']:.2f} ms")
    print(f"Aggregate throughput: {result['aggregate_gib_per_s']:.3f} GiB/s")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
