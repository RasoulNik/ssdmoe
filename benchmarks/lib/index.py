"""Model-agnostic constants from expert_index.json.

All per-model numbers (n_experts, n_moe_layers, top_k, bytes per expert) are
stored in the index built by tools/build_moe_index.py.  Scripts should read
from here rather than hardcoding 35B- or 122B-specific values.

Usage:
    from lib.index import load_index_config

    cfg = load_index_config(Path(args.index))
    print(cfg.n_experts)               # 256 for Qwen3.5, model-specific otherwise
    print(cfg.n_moe_layers)            # 40 for 35B, 48 for 122B, etc.
    print(cfg.ssd_ms_per_token(3.4))   # expected read time at 3.4 GB/s
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IndexConfig:
    """Immutable snapshot of per-model constants from expert_index.json."""

    model_type: str
    n_hidden_layers: int
    n_moe_layers: int
    n_experts: int
    n_experts_per_tok: int          # default top-K used during index build
    routed_bytes_per_token: int     # total bytes read per decode token at default top-K
    bytes_per_expert_per_layer: int  # routed_bytes / (n_moe_layers * top_k)

    def ssd_ms_per_token(self, bw_gbps: float, top_k: int | None = None) -> float:
        """Expected SSD read time in ms per decode token at the given bandwidth."""
        k = top_k if top_k is not None else self.n_experts_per_tok
        gb = self.n_moe_layers * k * self.bytes_per_expert_per_layer / 1e9
        return gb / bw_gbps * 1000.0

    def tps_at_bw(self, bw_gbps: float, top_k: int | None = None) -> float:
        """Theoretical max decode tok/s limited by SSD bandwidth."""
        ms = self.ssd_ms_per_token(bw_gbps, top_k)
        return 1000.0 / ms if ms > 0 else 0.0


def load_index_config(index_path: Path) -> IndexConfig:
    """Parse an expert_index.json produced by tools/build_moe_index.py."""
    with Path(index_path).open() as f:
        idx = json.load(f)
    cfg = idx.get("config", {})
    bytes_info = idx.get("bytes", {})

    n_moe = cfg.get("num_moe_layers", 0)
    top_k = cfg.get("num_experts_per_tok") or 0
    rbt = bytes_info.get("routed_bytes_per_token") or 0
    bpe = (rbt // (n_moe * top_k)) if (rbt and n_moe and top_k) else 0

    return IndexConfig(
        model_type=cfg.get("model_type", idx.get("model_type", "unknown")),
        n_hidden_layers=cfg.get("num_hidden_layers", 0),
        n_moe_layers=n_moe,
        n_experts=cfg.get("num_experts", 0),
        n_experts_per_tok=top_k,
        routed_bytes_per_token=rbt,
        bytes_per_expert_per_layer=bpe,
    )
