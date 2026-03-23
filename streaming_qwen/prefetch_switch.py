"""Prefetching StreamedSwitchGLU that overlaps I/O with compute.

This module implements a prefetch strategy where experts for the next
forward pass are loaded asynchronously while the current forward pass
is computing.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu

from .expert_store import ExpertStore
from .streamed_switch import (
    STREAM_STATS,
    _blob_to_mx,
    _component_to_mx,
    _remap_indices,
)


class PrefetchManager:
    """Manages prefetching of expert weights across layers."""

    def __init__(self, expert_store: ExpertStore, num_layers: int):
        self.expert_store = expert_store
        self.num_layers = num_layers
        self._executor = ThreadPoolExecutor(max_workers=4)
        # Keep only one speculative payload per layer. Same-layer next-token
        # reuse is the only hypothesis with meaningful overlap, so retaining
        # older expert sets just burns memory on this 16 GB machine.
        self._prefetch_cache: dict[int, tuple[frozenset[int], Future]] = {}
        self._lock = threading.Lock()
        self._stats = {
            "prefetch_hits": 0,
            "prefetch_misses": 0,
            "prefetch_submitted": 0,
        }

    def submit_prefetch(
        self, layer_idx: int, expert_indices: list[int], components: list[str]
    ) -> None:
        """Submit a prefetch request for the given experts."""
        expert_key = frozenset(expert_indices)
        with self._lock:
            existing = self._prefetch_cache.get(layer_idx)
            if existing is not None and existing[0] == expert_key:
                return
            if existing is not None:
                existing[1].cancel()
            self._stats["prefetch_submitted"] += 1
            future = self._executor.submit(
                self._do_prefetch, layer_idx, expert_indices, components
            )
            self._prefetch_cache[layer_idx] = (expert_key, future)

    def _do_prefetch(
        self, layer_idx: int, expert_indices: list[int], components: list[str]
    ) -> dict[str, memoryview]:
        """Actually load the experts (runs in background thread)."""
        return self.expert_store.read_components_batched(
            layer_idx, expert_indices, components=components
        )

    def get_prefetched(
        self, layer_idx: int, expert_indices: list[int], timeout: float = 0.0
    ) -> Optional[dict[str, memoryview]]:
        """Try to get prefetched experts. Returns None if not available."""
        expert_key = frozenset(expert_indices)
        with self._lock:
            cached = self._prefetch_cache.get(layer_idx)
            if cached is not None and cached[0] == expert_key:
                _, future = self._prefetch_cache.pop(layer_idx)
            else:
                future = None
        if future is None:
            self._stats["prefetch_misses"] += 1
            return None
        if future.done():
            self._stats["prefetch_hits"] += 1
            return future.result()
        if timeout > 0:
            try:
                result = future.result(timeout=timeout)
                self._stats["prefetch_hits"] += 1
                return result
            except Exception:
                self._stats["prefetch_misses"] += 1
                return None
        self._stats["prefetch_misses"] += 1
        return None

    def clear(self) -> None:
        """Clear all pending prefetches."""
        with self._lock:
            for _, future in self._prefetch_cache.values():
                future.cancel()
            self._prefetch_cache.clear()

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)

    @property
    def stats(self) -> dict:
        return dict(self._stats)


class PrefetchingStreamedSwitchGLU(nn.Module):
    """StreamedSwitchGLU with prefetching support.

    Key optimization: After computing routing for layer N, we submit a prefetch
    request for the same layer. On the next token, if the same experts are
    selected (common in autoregressive generation), the data is already loaded.
    """

    def __init__(
        self,
        layer_idx: int,
        expert_store: ExpertStore,
        prefetch_manager: Optional[PrefetchManager] = None,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.expert_store = expert_store
        self.prefetch_manager = prefetch_manager
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        # Track last selected experts for speculation
        self._last_selected: list[int] = []
        self._stats = {
            "prefetch_used": 0,
            "prefetch_missed": 0,
        }

    def _load_selected(self, selected_experts: list[int]) -> dict[str, mx.array]:
        """Load selected experts, using prefetch if available."""
        layer_info = self.expert_store.expert_reads[str(self.layer_idx)]
        streamed_components = list(layer_info.keys())

        # Try to get from prefetch cache
        payload = None
        if self.prefetch_manager is not None:
            payload = self.prefetch_manager.get_prefetched(
                self.layer_idx, selected_experts, timeout=0.001
            )
            if payload is not None:
                self._stats["prefetch_used"] += 1
            else:
                self._stats["prefetch_missed"] += 1

        # If not prefetched, load synchronously
        if payload is None:
            t0 = time.perf_counter()
            if self.expert_store.native_reader is not None:
                payload = self.expert_store.read_components_batched(
                    self.layer_idx,
                    selected_experts,
                    components=streamed_components,
                )
            else:
                payload = self.expert_store.read_experts_parallel(
                    self.layer_idx,
                    selected_experts,
                    max_workers=len(selected_experts),
                )
            STREAM_STATS["load_seconds"] += time.perf_counter() - t0

        # Submit prefetch for next time (same layer, same experts)
        # This helps when the same experts are selected across tokens
        if self.prefetch_manager is not None:
            self.prefetch_manager.submit_prefetch(
                self.layer_idx, selected_experts, streamed_components
            )

        # Convert to MLX arrays
        out = {}
        for component, info in layer_info.items():
            t1 = time.perf_counter()
            if self.expert_store.native_reader is not None:
                out[component] = _blob_to_mx(payload[component], info, len(selected_experts))
            else:
                batch_bytes = [payload[expert_idx][component] for expert_idx in selected_experts]
                out[component] = _component_to_mx(batch_bytes, info)
            STREAM_STATS["convert_seconds"] += time.perf_counter() - t1
        return out

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        STREAM_STATS["calls"] += 1
        selected = sorted(set(int(v) for v in np.array(indices.tolist()).flatten()))
        STREAM_STATS["selected_experts_total"] += len(selected)
        if not selected:
            return mx.zeros((*x.shape[:-1], 0, x.shape[-1]), dtype=x.dtype)

        t0 = time.perf_counter()
        local_indices = _remap_indices(indices, selected)
        STREAM_STATS["remap_seconds"] += time.perf_counter() - t0
        tensors = self._load_selected(selected)

        # Store for speculation
        self._last_selected = selected

        x = mx.expand_dims(x, (-2, -3))
        t1 = time.perf_counter()
        x_up = mx.gather_qmm(
            x,
            tensors["up_proj.weight"],
            tensors["up_proj.scales"],
            tensors["up_proj.biases"],
            rhs_indices=local_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=False,
        )
        STREAM_STATS["qmm_up_seconds"] += time.perf_counter() - t1

        t2 = time.perf_counter()
        x_gate = mx.gather_qmm(
            x,
            tensors["gate_proj.weight"],
            tensors["gate_proj.scales"],
            tensors["gate_proj.biases"],
            rhs_indices=local_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=False,
        )
        STREAM_STATS["qmm_gate_seconds"] += time.perf_counter() - t2

        t3 = time.perf_counter()
        activated = swiglu(x_gate, x_up)
        STREAM_STATS["swiglu_seconds"] += time.perf_counter() - t3

        t4 = time.perf_counter()
        x_down = mx.gather_qmm(
            activated,
            tensors["down_proj.weight"],
            tensors["down_proj.scales"],
            tensors["down_proj.biases"],
            rhs_indices=local_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=False,
        )
        STREAM_STATS["qmm_down_seconds"] += time.perf_counter() - t4
        return x_down.squeeze(-2)


def get_prefetch_stats(switches: list[PrefetchingStreamedSwitchGLU]) -> dict:
    """Aggregate prefetch stats from all switches."""
    total = {"prefetch_used": 0, "prefetch_missed": 0}
    for sw in switches:
        for k, v in sw._stats.items():
            total[k] = total.get(k, 0) + v
    return total
