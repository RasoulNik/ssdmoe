from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import _get_classes, load_config, load_tokenizer

from .expert_store import ExpertStore
from .model_io import (
    list_non_expert_text_tensors,
    load_expert_aux_weights,
    load_non_expert_text_weights,
)
from .prefetch_switch import PrefetchManager, PrefetchingStreamedSwitchGLU
from .streamed_switch import StreamedSwitchGLU


def iter_moe_layers(model):
    text_model = getattr(getattr(model, "language_model", model), "model", model)
    for layer in text_model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "top_k"):
            yield layer


def set_routed_top_k(model, top_k: int) -> None:
    for layer in iter_moe_layers(model):
        layer.mlp.top_k = top_k


def _patch_streamed_switches(
    model,
    expert_store: ExpertStore,
    quantization: dict,
    cache_limit_bytes: int = 0,
    use_prefetch: bool = False,
) -> None:
    layers = list(iter_moe_layers(model))
    prefetch_manager = (
        PrefetchManager(expert_store, num_layers=len(layers)) if use_prefetch else None
    )
    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "switch_mlp"):
            continue
        if use_prefetch:
            mlp.switch_mlp = PrefetchingStreamedSwitchGLU(
                layer_idx=layer_idx,
                expert_store=expert_store,
                prefetch_manager=prefetch_manager,
                group_size=quantization.get("group_size", 64),
                bits=quantization.get("bits", 4),
                mode=quantization.get("mode", "affine"),
            )
        else:
            mlp.switch_mlp = StreamedSwitchGLU(
                layer_idx=layer_idx,
                expert_store=expert_store,
                group_size=quantization.get("group_size", 64),
                bits=quantization.get("bits", 4),
                mode=quantization.get("mode", "affine"),
                cache_limit_bytes=cache_limit_bytes,
            )
    expert_store.prefetch_manager = prefetch_manager


def _quantize_resident_modules(model, config: dict, available_weight_names: set[str]) -> None:
    quantization = config.get("quantization") or config.get("quantization_config")
    if not quantization:
        return

    def class_predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        return f"{path}.scales" in available_weight_names

    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        mode=quantization.get("mode", "affine"),
        class_predicate=class_predicate,
    )


def build_streamed_model(
    model_path: Path,
    index_path: Path,
    top_k: int | None = None,
    cache_limit_bytes: int = 0,
    use_nocache: bool = False,
    native_reader_path: Path | None = None,
    resident_small_components: bool = False,
    component_workers: int = 3,
    use_prefetch: bool = False,
):
    model_path = Path(model_path).expanduser().resolve()
    index_path = Path(index_path).expanduser().resolve()

    config = load_config(model_path)
    model_class, model_args_class = _get_classes(config=config)
    model = model_class(model_args_class.from_dict(config))

    resident_components = (
        load_expert_aux_weights(model_path) if resident_small_components else None
    )
    expert_store = ExpertStore(
        index_path,
        use_nocache=use_nocache,
        native_reader_path=native_reader_path,
        resident_components=resident_components,
        component_workers=component_workers,
    )
    expert_store.open()

    available_weight_names = set(list_non_expert_text_tensors(model_path))
    _patch_streamed_switches(
        model,
        expert_store,
        config.get("quantization") or config.get("quantization_config") or {},
        cache_limit_bytes=cache_limit_bytes,
        use_prefetch=use_prefetch,
    )
    _quantize_resident_modules(model, config, available_weight_names)

    weights = load_non_expert_text_weights(model_path)
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)
    model.eval()
    mx.eval(model.parameters())

    if top_k is not None:
        set_routed_top_k(model, top_k)

    tokenizer = load_tokenizer(model_path)
    return model, tokenizer, expert_store, config
