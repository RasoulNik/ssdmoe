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
from .pipelined_moe import patch_pipelined_moe
from .prefetch_switch import PrefetchManager, PrefetchingStreamedSwitchGLU
from .session_window_cache import SessionWindowNativeCache
from .streamed_switch import StreamedSwitchGLU


class _LatentMoEWrapper(nn.Module):
    """Wraps NemotronHMoE for models with moe_latent_size (e.g. Nemotron-120B).

    mlx-lm's NemotronHMoE passes the full hidden state (4096-dim) to switch_mlp.
    The 120B model's experts operate in a latent space (1024-dim): the mixer owns
    fc1_latent_proj (hidden→latent) and fc2_latent_proj (latent→hidden) around them.

    After _patch_streamed_switches replaces switch_mlp with StreamedSwitchGLU, this
    wrapper is set as layer.mixer so that model.load_weights() correctly populates
    fc1_latent_proj and fc2_latent_proj via the standard path-based mechanism.
    """

    def __init__(self, original_moe, hidden_size: int, latent_size: int):
        super().__init__()
        self.switch_mlp = original_moe.switch_mlp      # StreamedSwitchGLU
        self.gate = original_moe.gate
        self.shared_experts = getattr(original_moe, "shared_experts", None)
        # Placeholder shapes — replaced by model.load_weights() with quantized weights
        self.fc1_latent_proj = nn.Linear(hidden_size, latent_size, bias=False)
        self.fc2_latent_proj = nn.Linear(latent_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        x_lat = self.fc1_latent_proj(x)                                # [B, latent]
        y = self.switch_mlp(x_lat, inds)                               # [B, K, latent]
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)       # [B, latent]
        y = self.fc2_latent_proj(y)                                    # [B, hidden]
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


def _wrap_latent_moe_layers(model, config: dict) -> None:
    """Replace NemotronHMoE mixers with _LatentMoEWrapper when moe_latent_size is set."""
    latent_size = config.get("moe_latent_size")
    if not latent_size:
        return
    hidden_size = config.get("hidden_size")
    text_model = getattr(getattr(model, "language_model", model), "model", model)
    for layer in text_model.layers:
        mixer = getattr(layer, "mixer", None)
        if mixer is not None and hasattr(mixer, "switch_mlp"):
            layer.mixer = _LatentMoEWrapper(mixer, hidden_size, latent_size)


def _get_moe_module(layer):
    """Return (moe_module, set_top_k) for a layer, or (None, None) if not a MoE layer.

    Handles two layouts:
      Qwen3:    layer.mlp.switch_mlp   / top_k on mlp
      Nemotron: layer.mixer.switch_mlp / top_k on mixer.gate
    """
    mlp = getattr(layer, "mlp", None)
    if mlp is not None and hasattr(mlp, "switch_mlp"):
        return mlp, lambda k: setattr(mlp, "top_k", k)
    mixer = getattr(layer, "mixer", None)
    if mixer is not None and hasattr(mixer, "switch_mlp"):
        gate = getattr(mixer, "gate", None)
        setter = (lambda k: setattr(gate, "top_k", k)) if gate is not None else (lambda k: None)
        return mixer, setter
    return None, None


def iter_moe_layers(model):
    text_model = getattr(getattr(model, "language_model", model), "model", model)
    for layer in text_model.layers:
        moe_module, _ = _get_moe_module(layer)
        if moe_module is not None:
            yield layer


def set_routed_top_k(model, top_k: int) -> None:
    for layer in iter_moe_layers(model):
        _, set_top_k = _get_moe_module(layer)
        if set_top_k is not None:
            set_top_k(top_k)


def _patch_streamed_switches(
    model,
    expert_store: ExpertStore,
    quantization: dict,
    cache_limit_bytes: int = 0,
    use_prefetch: bool = False,
    fused_gate_up: bool = False,
    compile_fused_gate_up: bool = False,
    session_cache: SessionWindowNativeCache | None = None,
    activation: str = "swiglu",
) -> None:
    # Use global layer indices so layer_idx matches the expert index keys.
    # For Qwen all layers are MoE so global_idx==moe_idx, but for hybrid
    # models (Nemotron) only some layers are MoE — we must use the real
    # position in text_model.layers (== the index in the safetensors key).
    text_model = getattr(getattr(model, "language_model", model), "model", model)
    all_layers = list(text_model.layers)
    moe_layer_pairs = [
        (i, layer) for i, layer in enumerate(all_layers)
        if _get_moe_module(layer)[0] is not None
    ]
    prefetch_manager = (
        PrefetchManager(expert_store, num_layers=len(moe_layer_pairs)) if use_prefetch else None
    )
    for layer_idx, layer in moe_layer_pairs:
        moe_module, _ = _get_moe_module(layer)
        if moe_module is None:
            continue
        if use_prefetch:
            moe_module.switch_mlp = PrefetchingStreamedSwitchGLU(
                layer_idx=layer_idx,
                expert_store=expert_store,
                prefetch_manager=prefetch_manager,
                group_size=quantization.get("group_size", 64),
                bits=quantization.get("bits", 4),
                mode=quantization.get("mode", "affine"),
                fused_gate_up=fused_gate_up,
                compile_fused_gate_up=compile_fused_gate_up,
            )
        else:
            moe_module.switch_mlp = StreamedSwitchGLU(
                layer_idx=layer_idx,
                expert_store=expert_store,
                group_size=quantization.get("group_size", 64),
                bits=quantization.get("bits", 4),
                mode=quantization.get("mode", "affine"),
                cache_limit_bytes=cache_limit_bytes,
                fused_gate_up=fused_gate_up,
                compile_fused_gate_up=compile_fused_gate_up,
                session_cache=session_cache,
                activation=activation,
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
    moe_impl: str = "streamed",
    fused_gate_up: bool = False,
    compile_fused_gate_up: bool = False,
    expert_cache_strategy: str = "none",
    expert_window_tokens: int = 0,
    expert_cache_bytes: int = 0,
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
    quantization = config.get("quantization") or config.get("quantization_config") or {}

    session_cache: SessionWindowNativeCache | None = None
    if expert_cache_strategy == "session_window_native" and expert_window_tokens > 0:
        session_cache = SessionWindowNativeCache(
            max_bytes=expert_cache_bytes,
            window_tokens=expert_window_tokens,
        )

    # Detect activation from model type: Nemotron uses relu2 (no gate, 2-component experts)
    model_type = config.get("model_type", "")
    activation = "relu2" if "nemotron" in model_type.lower() else "swiglu"

    if moe_impl != "pipelined":
        _patch_streamed_switches(
            model,
            expert_store,
            quantization,
            cache_limit_bytes=cache_limit_bytes,
            use_prefetch=use_prefetch,
            fused_gate_up=fused_gate_up,
            compile_fused_gate_up=compile_fused_gate_up,
            session_cache=session_cache,
            activation=activation,
        )
        # For models with moe_latent_size (e.g. Nemotron-120B): inject latent projection
        # wrappers so that fc1_latent_proj / fc2_latent_proj weights are registered as
        # proper module paths and loaded correctly by model.load_weights().
        _wrap_latent_moe_layers(model, config)
    _quantize_resident_modules(model, config, available_weight_names)

    weights = load_non_expert_text_weights(model_path)
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)
    model.eval()
    mx.eval(model.parameters())

    if moe_impl == "pipelined":
        patch_pipelined_moe(
            model,
            expert_store,
            quantization,
            fused_gate_up=fused_gate_up,
            compile_fused_gate_up=compile_fused_gate_up,
        )

    if top_k is not None:
        set_routed_top_k(model, top_k)

    tokenizer = load_tokenizer(model_path)
    return model, tokenizer, expert_store, config


# ---------------------------------------------------------------------------
# Session-window cache helpers (called per-request by server / bench scripts)
# ---------------------------------------------------------------------------

def _get_session_cache(model) -> SessionWindowNativeCache | None:
    """Return the shared SessionWindowNativeCache from the first MoE layer, or None."""
    for layer in iter_moe_layers(model):
        moe_module, _ = _get_moe_module(layer)
        if moe_module is not None:
            switch = getattr(moe_module, "switch_mlp", None)
            if switch is not None:
                return getattr(switch, "session_cache", None)
    return None


def begin_session_cache_request(
    model,
    *,
    session_id: str | None,
    phase: str,
    enabled: bool,
    ephemeral: bool,
) -> None:
    cache = _get_session_cache(model)
    if cache is not None:
        cache.begin_request(
            session_id=session_id,
            phase=phase,
            enabled=enabled,
            ephemeral=ephemeral,
        )


def end_session_cache_request(model) -> None:
    cache = _get_session_cache(model)
    if cache is not None:
        cache.end_request()


def set_session_cache_phase(model, phase: str) -> None:
    cache = _get_session_cache(model)
    if cache is not None:
        cache.set_phase(phase)


def complete_session_cache_token(model) -> None:
    cache = _get_session_cache(model)
    if cache is not None:
        cache.complete_token()


def collect_session_cache_stats(model) -> dict | None:
    cache = _get_session_cache(model)
    if cache is None:
        return None
    return cache.stats()


def collect_window_cache_stats(model) -> dict:
    """Aggregate stats from the in-process LRU tensor cache (cache_limit_bytes path)."""
    entries = 0
    bytes_used = 0
    for layer in iter_moe_layers(model):
        moe_module, _ = _get_moe_module(layer)
        switch = getattr(moe_module, "switch_mlp", None) if moe_module is not None else None
        if switch is not None:
            entries += len(getattr(switch, "_cache", {}))
            bytes_used += getattr(switch, "_cache_bytes", 0)
    return {
        "current_entries": entries,
        "current_bytes": bytes_used,
        "peak_entries": 0,
        "peak_bytes": 0,
    }


def set_window_cache_enabled(model, enabled: bool, *, reset: bool = False) -> None:
    """Enable or disable the in-process LRU tensor cache on all MoE layers."""
    for layer in iter_moe_layers(model):
        moe_module, _ = _get_moe_module(layer)
        switch = getattr(moe_module, "switch_mlp", None) if moe_module is not None else None
        if switch is None:
            continue
        if reset and hasattr(switch, "_cache"):
            switch._cache.clear()
            switch._cache_bytes = 0
