"""Pipelined MoE block that overlaps I/O with shared expert compute.

Key insight: In SparseMoeBlock, after getting routing indices, we:
1. Load experts (I/O)
2. Compute with experts (GPU)
3. Compute shared_expert (GPU)

By reordering and parallelizing:
1. Get routing indices
2. Start expert I/O (background thread)
3. Compute shared_expert (GPU) - overlaps with I/O!
4. Wait for I/O
5. Compute with loaded experts (GPU)
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients
from .expert_store import ExpertStore
from .streamed_switch import (
    STREAM_STATS,
    PROFILE_STREAMED,
    _blob_to_mx,
    _compute_expert_output,
    _remap_indices,
)


class PipelinedStreamedSwitchGLU(nn.Module):
    """StreamedSwitchGLU with async I/O support for pipelining."""

    def __init__(
        self,
        layer_idx: int,
        expert_store: ExpertStore,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        fused_gate_up: bool = False,
        compile_fused_gate_up: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.expert_store = expert_store
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.fused_gate_up = fused_gate_up
        self.compile_fused_gate_up = compile_fused_gate_up
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_load: Optional[Future] = None
        self._pending_experts: Optional[list[int]] = None

    def start_async_load(self, indices: mx.array) -> tuple[list[int], mx.array]:
        """Start loading experts asynchronously. Returns (selected, local_indices)."""
        selected = sorted(set(int(v) for v in np.array(indices.tolist()).flatten()))
        if not selected:
            return [], mx.array([])

        local_indices = _remap_indices(indices, selected)
        layer_info = self.expert_store.expert_reads[str(self.layer_idx)]
        components = list(layer_info.keys())

        # Submit async load
        self._pending_experts = selected
        self._pending_load = self._executor.submit(
            self.expert_store.read_components_batched,
            self.layer_idx,
            selected,
            components,
        )
        return selected, local_indices

    def wait_and_compute(
        self,
        x: mx.array,
        selected: list[int],
        local_indices: mx.array,
    ) -> mx.array:
        """Wait for async load and compute the expert output."""
        if not selected:
            return mx.zeros((*x.shape[:-1], 0, x.shape[-1]), dtype=x.dtype)

        STREAM_STATS["calls"] += 1
        STREAM_STATS["selected_experts_total"] += len(selected)

        # Wait for I/O
        t0 = time.perf_counter()
        payload = self._pending_load.result()
        STREAM_STATS["load_seconds"] += time.perf_counter() - t0

        # Convert to MLX arrays
        layer_info = self.expert_store.expert_reads[str(self.layer_idx)]
        tensors = {}
        for component, info in layer_info.items():
            t1 = time.perf_counter()
            tensors[component] = _blob_to_mx(payload[component], info, len(selected))
            STREAM_STATS["convert_seconds"] += time.perf_counter() - t1

        self._pending_load = None
        self._pending_experts = None
        return _compute_expert_output(
            x,
            tensors,
            local_indices,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            fused_gate_up=self.fused_gate_up,
            compile_fused_gate_up=self.compile_fused_gate_up,
        )

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        """Standard call interface (not pipelined)."""
        selected, local_indices = self.start_async_load(indices)
        return self.wait_and_compute(x, selected, local_indices)

    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)


class PipelinedSparseMoeBlock(nn.Module):
    """SparseMoeBlock with pipelined I/O.

    The key optimization: overlap expert I/O with shared_expert compute.
    """

    def __init__(
        self,
        original_moe_block,
        layer_idx: int,
        expert_store: ExpertStore,
        quantization: dict,
        fused_gate_up: bool = False,
        compile_fused_gate_up: bool = False,
    ):
        super().__init__()
        # Copy attributes from original
        self.norm_topk_prob = original_moe_block.norm_topk_prob
        self.num_experts = original_moe_block.num_experts
        self.top_k = original_moe_block.top_k

        # Reuse original modules (they may be quantized)
        self.gate = original_moe_block.gate
        self.shared_expert = original_moe_block.shared_expert
        self.shared_expert_gate = original_moe_block.shared_expert_gate

        # Replace switch_mlp with pipelined version
        self.switch_mlp = PipelinedStreamedSwitchGLU(
            layer_idx=layer_idx,
            expert_store=expert_store,
            group_size=quantization.get("group_size", 64),
            bits=quantization.get("bits", 4),
            mode=quantization.get("mode", "affine"),
            fused_gate_up=fused_gate_up,
            compile_fused_gate_up=compile_fused_gate_up,
        )

        self.sharding_group = getattr(original_moe_block, "sharding_group", None)

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        # Compute routing
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        # PIPELINED: Start expert I/O asynchronously
        selected, local_indices = self.switch_mlp.start_async_load(inds)

        # PIPELINED: Compute shared_expert while I/O is running
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        mx.eval(shared_y)  # Force evaluation to overlap with I/O

        # PIPELINED: Wait for I/O and compute expert output
        y = self.switch_mlp.wait_and_compute(x, selected, local_indices)
        y = (y * scores[..., None]).sum(axis=-2)

        y = y + shared_y

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


def patch_pipelined_moe(
    model,
    expert_store: ExpertStore,
    quantization: dict,
    fused_gate_up: bool = False,
    compile_fused_gate_up: bool = False,
) -> None:
    """Replace SparseMoeBlock with PipelinedSparseMoeBlock in the model."""
    patched_count = 0
    for layer_idx, layer in enumerate(model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        # Check if this is a SparseMoeBlock (has switch_mlp)
        if not hasattr(mlp, "switch_mlp"):
            continue

        # Create pipelined version that wraps the original
        pipelined = PipelinedSparseMoeBlock(
            original_moe_block=mlp,
            layer_idx=layer_idx,
            expert_store=expert_store,
            quantization=quantization,
            fused_gate_up=fused_gate_up,
            compile_fused_gate_up=compile_fused_gate_up,
        )

        layer.mlp = pipelined
        patched_count += 1

    print(f"Patched {patched_count} MoE layers with pipelined I/O")


PIPELINED_STATS = {
    "shared_expert_overlap_ms": 0.0,
}


def reset_pipelined_stats():
    for key in PIPELINED_STATS:
        PIPELINED_STATS[key] = 0.0
