"""Expert selection recording hooks for MoE benchmarks.

Provides a reusable RecordingSwitch wrapper and install_recording_hooks()
that replaces every StreamedSwitchGLU in the model with a thin recording
wrapper that captures the frozenset of activated expert indices per token.

Usage:
    from collections import defaultdict
    from lib.hooks import install_recording_hooks

    selections = defaultdict(list)          # layer_idx → [frozenset, ...]
    patched = install_recording_hooks(model, selections)
    # ... run decode ...
    # selections[layer_idx] now contains one frozenset per token
"""
from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from lib.loader import ensure_src_path

ensure_src_path()

from streaming_moe.streamed_switch import StreamedSwitchGLU
from streaming_moe.prefetch_switch import PrefetchingStreamedSwitchGLU


class RecordingSwitch(nn.Module):
    """Wraps a StreamedSwitchGLU and records activated expert indices per call.

    The frozenset of unique expert indices for each forward call is appended to
    record_dict[layer_idx].  The real switch is then called unchanged so all
    existing model behaviour is preserved.

    Args:
        inner:       The real StreamedSwitchGLU to delegate to.
        layer_idx:   Key used in record_dict.
        record_dict: Mutable dict; appended in-place.
    """

    def __init__(
        self,
        inner: StreamedSwitchGLU,
        layer_idx: int,
        record_dict: DefaultDict[int, list],
    ) -> None:
        super().__init__()
        self._inner = inner
        self._layer_idx = layer_idx
        self._record = record_dict

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        indices_np = np.array(indices.tolist(), dtype=np.int32)
        selected = frozenset(np.unique(indices_np).tolist())
        self._record[self._layer_idx].append(selected)
        return self._inner(x, indices)


def install_recording_hooks(
    model,
    record_dict: DefaultDict[int, list],
) -> list[int]:
    """Replace every StreamedSwitchGLU in model with a RecordingSwitch.

    Returns the sorted list of MoE layer indices that were patched.
    Idempotent only if called once; calling twice will double-wrap.
    """
    from streaming_moe.runtime import _get_moe_module

    text_model = getattr(getattr(model, "language_model", model), "model", model)
    patched: list[int] = []
    for i, layer in enumerate(text_model.layers):
        moe_module, _ = _get_moe_module(layer)
        if moe_module is None:
            continue
        switch = getattr(moe_module, "switch_mlp", None)
        if switch is None:
            continue
        if isinstance(switch, (StreamedSwitchGLU, PrefetchingStreamedSwitchGLU)):
            moe_module.switch_mlp = RecordingSwitch(switch, i, record_dict)
            patched.append(i)
    return patched
