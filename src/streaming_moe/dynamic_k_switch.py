"""Dynamic K expert selection based on routing scores.

Instead of always using the same K experts, dynamically select based on
routing score thresholds. If one expert has a very high score, we might
only need to load that one.
"""
from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu

from .expert_store import ExpertStore
from .streamed_switch import (
    STREAM_STATS,
    PROFILE_STREAMED,
    _blob_to_mx,
    _remap_indices,
)


class DynamicKStreamedSwitchGLU(nn.Module):
    """StreamedSwitchGLU with dynamic K based on routing scores.

    Instead of always loading K experts, we dynamically decide based on:
    1. Score threshold: Only load experts above a minimum score
    2. Cumulative threshold: Stop when cumulative score reaches a target
    3. Minimum K: Always load at least min_k experts

    This allows faster inference when routing is confident.
    """

    def __init__(
        self,
        layer_idx: int,
        expert_store: ExpertStore,
        max_k: int = 8,
        min_k: int = 1,
        score_threshold: float = 0.05,  # Minimum score to include expert
        cumulative_threshold: float = 0.95,  # Stop when cumulative >= this
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.expert_store = expert_store
        self.max_k = max_k
        self.min_k = min_k
        self.score_threshold = score_threshold
        self.cumulative_threshold = cumulative_threshold
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self._dynamic_k_stats = {
            "calls": 0,
            "total_k": 0,
            "k_distribution": {},
        }

    def _select_experts_dynamic(
        self, gates: mx.array, original_k: int
    ) -> tuple[mx.array, mx.array, int]:
        """Dynamically select experts based on scores.

        Returns (indices, scores, actual_k).
        """
        # Get all gates for analysis
        gates_np = np.array(gates.tolist())

        # Get top max_k candidates
        top_indices = np.argpartition(gates_np, -self.max_k, axis=-1)[..., -self.max_k:]
        top_scores = np.take_along_axis(gates_np, top_indices, axis=-1)

        # Sort by score (descending)
        sort_order = np.argsort(-top_scores, axis=-1)
        top_indices = np.take_along_axis(top_indices, sort_order, axis=-1)
        top_scores = np.take_along_axis(top_scores, sort_order, axis=-1)

        # Determine actual K based on thresholds
        # For now, use batch-level decision (same K for all positions)
        batch_scores = top_scores.flatten() if top_scores.ndim > 1 else top_scores

        actual_k = self.min_k
        cumulative = 0.0
        for i in range(min(self.max_k, original_k)):
            if i < self.min_k:
                cumulative += batch_scores[i] if len(batch_scores) > i else 0
                continue
            score = batch_scores[i] if len(batch_scores) > i else 0
            if score < self.score_threshold:
                break
            cumulative += score
            actual_k = i + 1
            if cumulative >= self.cumulative_threshold:
                break

        # Track statistics
        self._dynamic_k_stats["calls"] += 1
        self._dynamic_k_stats["total_k"] += actual_k
        k_str = str(actual_k)
        self._dynamic_k_stats["k_distribution"][k_str] = (
            self._dynamic_k_stats["k_distribution"].get(k_str, 0) + 1
        )

        # Return only actual_k experts
        selected_indices = top_indices[..., :actual_k]
        selected_scores = top_scores[..., :actual_k]

        return (
            mx.array(selected_indices.astype(np.int32)),
            mx.array(selected_scores),
            actual_k,
        )

    def _load_selected(self, selected_experts: list[int]) -> dict[str, mx.array]:
        layer_info = self.expert_store.expert_reads[str(self.layer_idx)]
        streamed_components = list(layer_info.keys())

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

        out = {}
        for component, info in layer_info.items():
            t1 = time.perf_counter()
            if self.expert_store.native_reader is not None:
                out[component] = _blob_to_mx(payload[component], info, len(selected_experts))
            else:
                batch_bytes = [payload[expert_idx][component] for expert_idx in selected_experts]
                from .streamed_switch import _component_to_mx
                out[component] = _component_to_mx(batch_bytes, info)
            STREAM_STATS["convert_seconds"] += time.perf_counter() - t1
        return out

    def __call__(
        self, x: mx.array, indices: mx.array, scores: mx.array = None
    ) -> mx.array:
        """Forward pass with dynamic K selection.

        Note: This expects to be called with pre-computed indices and scores
        from the SparseMoeBlock, not raw input.
        """
        STREAM_STATS["calls"] += 1

        # Get unique experts
        selected = sorted(set(int(v) for v in np.array(indices.tolist()).flatten()))
        STREAM_STATS["selected_experts_total"] += len(selected)

        if not selected:
            return mx.zeros((*x.shape[:-1], 0, x.shape[-1]), dtype=x.dtype)

        t0 = time.perf_counter()
        local_indices = _remap_indices(indices, selected)
        STREAM_STATS["remap_seconds"] += time.perf_counter() - t0

        tensors = self._load_selected(selected)

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
        if PROFILE_STREAMED:
            mx.eval(x_up)
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
        if PROFILE_STREAMED:
            mx.eval(x_gate)
        STREAM_STATS["qmm_gate_seconds"] += time.perf_counter() - t2

        t3 = time.perf_counter()
        activated = swiglu(x_gate, x_up)
        if PROFILE_STREAMED:
            mx.eval(activated)
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
        if PROFILE_STREAMED:
            mx.eval(x_down)
        STREAM_STATS["qmm_down_seconds"] += time.perf_counter() - t4

        return x_down.squeeze(-2)

    @property
    def dynamic_k_stats(self) -> dict:
        stats = dict(self._dynamic_k_stats)
        if stats["calls"] > 0:
            stats["avg_k"] = stats["total_k"] / stats["calls"]
        return stats


def get_dynamic_k_stats(switches: list[DynamicKStreamedSwitchGLU]) -> dict:
    """Aggregate dynamic K stats from all switches."""
    total = {
        "calls": 0,
        "total_k": 0,
        "k_distribution": {},
    }
    for sw in switches:
        stats = sw._dynamic_k_stats
        total["calls"] += stats["calls"]
        total["total_k"] += stats["total_k"]
        for k, count in stats["k_distribution"].items():
            total["k_distribution"][k] = total["k_distribution"].get(k, 0) + count
    if total["calls"] > 0:
        total["avg_k"] = total["total_k"] / total["calls"]
    return total
