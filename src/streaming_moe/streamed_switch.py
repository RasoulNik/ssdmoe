from __future__ import annotations

import time
from collections import OrderedDict
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu

from .expert_store import ExpertStore


STREAM_STATS = {
    "calls": 0,
    "selected_experts_total": 0,
    "remap_seconds": 0.0,
    "load_seconds": 0.0,
    "convert_seconds": 0.0,
    "qmm_up_seconds": 0.0,
    "qmm_gate_seconds": 0.0,
    "swiglu_seconds": 0.0,
    "qmm_down_seconds": 0.0,
    "eval_seconds": 0.0,
    "cache_hits": 0,
    "cache_misses": 0,
    "cache_evictions": 0,
}
PROFILE_STREAMED = False


def reset_stream_stats() -> None:
    for key in STREAM_STATS:
        STREAM_STATS[key] = 0 if key in {"calls", "selected_experts_total"} else 0.0


def set_stream_profiling(enabled: bool) -> None:
    global PROFILE_STREAMED
    PROFILE_STREAMED = enabled


def _component_to_mx(batch_bytes: list[bytes], info: dict) -> mx.array:
    component_shape = info["shape"][1:]
    dtype = info["dtype"]

    if dtype == "BF16":
        raw = np.frombuffer(b"".join(batch_bytes), dtype=np.uint16)
        raw = raw.reshape((len(batch_bytes), *component_shape))
        return mx.view(mx.asarray(raw), mx.bfloat16)

    if dtype == "U32":
        raw = np.frombuffer(b"".join(batch_bytes), dtype=np.uint32)
        raw = raw.reshape((len(batch_bytes), *component_shape))
        return mx.asarray(raw)

    raise ValueError(f"Unsupported streamed dtype: {dtype}")


def _blob_to_mx(blob, info: dict, count: int) -> mx.array:
    component_shape = (count, *info["shape"][1:])
    dtype = info["dtype"]
    if dtype == "BF16":
        raw = np.frombuffer(blob, dtype=np.uint16).reshape(component_shape)
        return mx.view(mx.asarray(raw), mx.bfloat16)
    if dtype == "U32":
        raw = np.frombuffer(blob, dtype=np.uint32).reshape(component_shape)
        return mx.asarray(raw)
    raise ValueError(f"Unsupported streamed dtype: {dtype}")


def _remap_indices(indices_np: np.ndarray, selected_np: np.ndarray) -> mx.array:
    # searchsorted on a sorted K-element array — pure numpy, no Python dispatch per element
    remapped = np.searchsorted(selected_np, indices_np).astype(np.int32)
    return mx.asarray(remapped)


def _compute_expert_output(
    x: mx.array,
    tensors: dict[str, mx.array],
    local_indices: mx.array,
    *,
    group_size: int,
    bits: int,
    mode: str,
    activation: str = "swiglu",
    fused_gate_up: bool = False,
    compile_fused_gate_up: bool = False,
) -> mx.array:
    if activation == "relu2":
        # Nemotron: fc1 → relu² → fc2  (no gate projection)
        x = mx.expand_dims(x, (-2, -3))
        t1 = time.perf_counter()
        x_up = mx.gather_qmm(
            x,
            tensors["fc1.weight"],
            tensors["fc1.scales"],
            tensors["fc1.biases"],
            rhs_indices=local_indices,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
            sorted_indices=False,
        )
        if PROFILE_STREAMED:
            mx.eval(x_up)
        STREAM_STATS["qmm_up_seconds"] += time.perf_counter() - t1

        t3 = time.perf_counter()
        activated = nn.relu(x_up) ** 2
        if PROFILE_STREAMED:
            mx.eval(activated)
        STREAM_STATS["swiglu_seconds"] += time.perf_counter() - t3

        t4 = time.perf_counter()
        x_down = mx.gather_qmm(
            activated,
            tensors["fc2.weight"],
            tensors["fc2.scales"],
            tensors["fc2.biases"],
            rhs_indices=local_indices,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
            sorted_indices=False,
        )
        if PROFILE_STREAMED:
            mx.eval(x_down)
        STREAM_STATS["qmm_down_seconds"] += time.perf_counter() - t4
        return x_down.squeeze(-2)

    if fused_gate_up:
        from .fused_expert import compute_expert_output_fused

        return compute_expert_output_fused(
            x,
            tensors,
            local_indices,
            group_size=group_size,
            bits=bits,
            mode=mode,
            use_compiled=compile_fused_gate_up,
        )

    x = mx.expand_dims(x, (-2, -3))
    t1 = time.perf_counter()
    x_up = mx.gather_qmm(
        x,
        tensors["up_proj.weight"],
        tensors["up_proj.scales"],
        tensors["up_proj.biases"],
        rhs_indices=local_indices,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
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
        group_size=group_size,
        bits=bits,
        mode=mode,
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
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=False,
    )
    if PROFILE_STREAMED:
        mx.eval(x_down)
    STREAM_STATS["qmm_down_seconds"] += time.perf_counter() - t4
    return x_down.squeeze(-2)


class StreamedSwitchGLU(nn.Module):
    """Flash-moe-style routed expert block backed by on-demand `pread()`."""

    def __init__(
        self,
        layer_idx: int,
        expert_store: ExpertStore,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        cache_limit_bytes: int = 0,
        fused_gate_up: bool = False,
        compile_fused_gate_up: bool = False,
        session_cache=None,
        activation: str = "swiglu",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.expert_store = expert_store
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.cache_limit_bytes = cache_limit_bytes
        self.fused_gate_up = fused_gate_up
        self.compile_fused_gate_up = compile_fused_gate_up
        self.session_cache = session_cache  # SessionWindowNativeCache | None
        self.activation = activation  # "swiglu" | "relu2"
        self._cache: OrderedDict[tuple[int, int], tuple[dict[str, mx.array], int]] = OrderedDict()
        self._cache_bytes = 0
        # Pre-compute per-layer constants so __call__ doesn't rebuild them every token
        self._layer_info = expert_store.expert_reads[str(layer_idx)]
        self._streamed_components = [
            c for c in self._layer_info
            if not expert_store.has_resident_component(layer_idx, c)
        ]

    def _cache_get(self, expert_idx: int) -> dict[str, mx.array] | None:
        key = (self.layer_idx, expert_idx)
        value = self._cache.get(key)
        if value is None:
            STREAM_STATS["cache_misses"] += 1
            return None
        self._cache.move_to_end(key)
        STREAM_STATS["cache_hits"] += 1
        return value[0]

    def _cache_put(self, expert_idx: int, tensors: dict[str, mx.array], size_bytes: int) -> None:
        if self.cache_limit_bytes <= 0:
            return
        key = (self.layer_idx, expert_idx)
        if key in self._cache:
            _, old_size = self._cache.pop(key)
            self._cache_bytes -= old_size
        self._cache[key] = (tensors, size_bytes)
        self._cache_bytes += size_bytes
        while self._cache_bytes > self.cache_limit_bytes and self._cache:
            _, (_, evicted_size) = self._cache.popitem(last=False)
            self._cache_bytes -= evicted_size
            STREAM_STATS["cache_evictions"] += 1

    def _load_single_expert(self, expert_idx: int) -> tuple[dict[str, mx.array], int]:
        cached = self._cache_get(expert_idx)
        if cached is not None:
            size_bytes = sum(arr.nbytes for arr in cached.values())
            return cached, size_bytes

        payload = self.expert_store.read_expert(self.layer_idx, expert_idx)
        layer_info = self.expert_store.expert_reads[str(self.layer_idx)]
        tensors = {}
        size_bytes = 0
        for component, info in layer_info.items():
            t1 = time.perf_counter()
            tensors[component] = _component_to_mx([payload[component]], info)[0]
            STREAM_STATS["convert_seconds"] += time.perf_counter() - t1
            size_bytes += tensors[component].nbytes
        self._cache_put(expert_idx, tensors, size_bytes)
        return tensors, size_bytes

    def _load_selected(self, selected_experts: list[int]) -> dict[str, mx.array]:
        layer_info = self._layer_info
        streamed_components = self._streamed_components

        # Session-window cache path: returns dict[str, mx.array] directly.
        # On hit: zero copy, zero alloc — returns existing cached mx.arrays.
        # On miss: reads from SSD and converts inside load_components_for_layer.
        if self.session_cache is not None:
            if streamed_components:
                result = self.session_cache.load_components_for_layer(
                    layer_idx=self.layer_idx,
                    selected_experts=selected_experts,
                    layer_info=layer_info,
                    expert_store=self.expert_store,
                    streamed_components=streamed_components,
                )
                if result is not None:
                    return result

        if self.cache_limit_bytes <= 0:
            selected_indices = mx.array(selected_experts, dtype=mx.int32)
            if self.expert_store.native_reader is not None:
                if streamed_components:
                    t0 = time.perf_counter()
                    payload = self.expert_store.read_components_batched(
                        self.layer_idx,
                        selected_experts,
                        components=streamed_components,
                    )
                    STREAM_STATS["load_seconds"] += time.perf_counter() - t0
                else:
                    payload = {}
            else:
                t0 = time.perf_counter()
                payload = self.expert_store.read_experts_parallel(
                    self.layer_idx,
                    selected_experts,
                    max_workers=len(selected_experts),
                )
                STREAM_STATS["load_seconds"] += time.perf_counter() - t0
            out = {}
            for component, info in layer_info.items():
                if self.expert_store.has_resident_component(self.layer_idx, component):
                    out[component] = mx.take(
                        self.expert_store.get_resident_component(self.layer_idx, component),
                        selected_indices,
                        axis=0,
                    )
                    continue
                t1 = time.perf_counter()
                if self.expert_store.native_reader is not None:
                    out[component] = _blob_to_mx(payload[component], info, len(selected_experts))
                else:
                    batch_bytes = [payload[expert_idx][component] for expert_idx in selected_experts]
                    out[component] = _component_to_mx(batch_bytes, info)
                STREAM_STATS["convert_seconds"] += time.perf_counter() - t1
            return out

        cached_payload: dict[int, dict[str, mx.array]] = {}
        missing = []
        for expert_idx in selected_experts:
            cached = self._cache_get(expert_idx)
            if cached is not None:
                cached_payload[expert_idx] = cached
            else:
                missing.append(expert_idx)

        if missing:
            t0 = time.perf_counter()
            payload = self.expert_store.read_experts_parallel(
                self.layer_idx,
                missing,
                max_workers=len(missing),
            )
            STREAM_STATS["load_seconds"] += time.perf_counter() - t0
            for expert_idx in missing:
                tensors = {}
                size_bytes = 0
                for component, info in layer_info.items():
                    t1 = time.perf_counter()
                    tensors[component] = _component_to_mx([payload[expert_idx][component]], info)[0]
                    STREAM_STATS["convert_seconds"] += time.perf_counter() - t1
                    size_bytes += tensors[component].nbytes
                cached_payload[expert_idx] = tensors
                self._cache_put(expert_idx, tensors, size_bytes)

        out = {}
        for component in layer_info.keys():
            out[component] = mx.stack(
                [cached_payload[expert_idx][component] for expert_idx in selected_experts],
                axis=0,
            )
        return out

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        STREAM_STATS["calls"] += 1
        # Single GPU→CPU sync for the whole layer-call.
        # np.unique returns a sorted array of distinct expert ids.
        t0 = time.perf_counter()
        indices_np = np.array(indices.tolist(), dtype=np.int32)
        selected_np = np.unique(indices_np)          # sorted, no Python loop
        selected = selected_np.tolist()
        STREAM_STATS["selected_experts_total"] += len(selected)
        if not selected:
            return mx.zeros((*x.shape[:-1], 0, x.shape[-1]), dtype=x.dtype)
        # Remap: searchsorted on the K-element sorted selected_np — pure numpy
        local_indices = _remap_indices(indices_np, selected_np)
        STREAM_STATS["remap_seconds"] += time.perf_counter() - t0
        tensors = self._load_selected(selected)
        return _compute_expert_output(
            x,
            tensors,
            local_indices,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            activation=self.activation,
            fused_gate_up=self.fused_gate_up,
            compile_fused_gate_up=self.compile_fused_gate_up,
        )
