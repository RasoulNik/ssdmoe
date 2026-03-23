#!/usr/bin/env python3
"""Benchmark flash-moe-style expert reads against a local expert index."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from streaming_qwen.expert_store import ExpertStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark routed expert pread()")
    parser.add_argument("--index", required=True, help="Path to expert_index.json")
    parser.add_argument("--layer", type=int, default=0, help="Layer to benchmark")
    parser.add_argument(
        "--experts",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated expert ids to read",
    )
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count (default: number of experts)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expert_ids = [int(x) for x in args.experts.split(",") if x.strip()]

    samples = []
    total_bytes = 0

    with ExpertStore(Path(args.index)) as store:
        for _ in range(args.iters):
            t0 = time.perf_counter()
            payload = store.read_experts_parallel(
                args.layer, expert_ids, max_workers=args.workers
            )
            elapsed = time.perf_counter() - t0
            samples.append(elapsed)
            if not total_bytes:
                total_bytes = sum(
                    len(component_bytes)
                    for expert in payload.values()
                    for component_bytes in expert.values()
                )

    mean_s = statistics.mean(samples)
    p50_s = statistics.median(samples)
    best_s = min(samples)
    gib = total_bytes / 1024**3

    print(f"Layer: {args.layer}")
    print(f"Experts: {expert_ids}")
    print(f"Payload: {gib:.4f} GiB per iteration")
    print(f"Mean: {mean_s * 1000:.2f} ms  ({gib / mean_s:.3f} GiB/s)")
    print(f"P50:  {p50_s * 1000:.2f} ms  ({gib / p50_s:.3f} GiB/s)")
    print(f"Best: {best_s * 1000:.2f} ms  ({gib / best_s:.3f} GiB/s)")


if __name__ == "__main__":
    main()
