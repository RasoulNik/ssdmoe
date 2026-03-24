#!/usr/bin/env python3
"""Build a routed-expert read index for any supported MLX MoE checkpoint.

Supports:
  Qwen3 MoE  (model_type: qwen3_moe)
    language_model.model.layers.{layer}.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.*
  Nemotron-H (model_type: nemotron_h)
    backbone.layers.{layer}.mixer.switch_mlp.{fc1,fc2}.*

Usage:
  poetry run python tools/build_moe_index.py \\
    --model ~/.cache/huggingface/hub/models--mlx-community--NVIDIA-Nemotron-3-Nano-30B-A3B-4bit/snapshots/<hash> \\
    --output .run/nemotron30b-expert-index.json
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Per-model-type expert tensor patterns
# ---------------------------------------------------------------------------

MODEL_EXPERT_RES: dict[str, re.Pattern] = {
    "qwen3_moe": re.compile(
        r"^(language_model\.)?model\.layers\.(?P<layer>\d+)\.mlp\.switch_mlp\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\."
        r"(?P<kind>weight|scales|biases)$"
    ),
    "nemotron_h": re.compile(
        r"^backbone\.layers\.(?P<layer>\d+)\.mixer\.switch_mlp\."
        r"(?P<proj>fc1|fc2)\."
        r"(?P<kind>weight|scales|biases)$"
    ),
}
VISION_RE = re.compile(r"^(vision_tower|model\.visual|visual)")


def detect_model_type(config: dict) -> str:
    mt = config.get("model_type", "")
    for known in MODEL_EXPERT_RES:
        if known in mt.lower():
            return known
    # Fallback: try Qwen pattern (covers older qwen2_moe etc.)
    return "qwen3_moe"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build routed-expert byte-offset index for any supported MLX MoE model"
    )
    p.add_argument("--model", required=True, help="Path to the local HF snapshot directory")
    p.add_argument("--output", default="expert_index.json", help="Output JSON path")
    p.add_argument(
        "--disk-gbps", type=float, default=None,
        help="Optional measured disk throughput in GB/s for tok/s estimates",
    )
    return p.parse_args()


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def build_index(model_path: Path) -> dict:
    with (model_path / "config.json").open() as f:
        config = json.load(f)
    with (model_path / "model.safetensors.index.json").open() as f:
        weight_map = json.load(f)["weight_map"]

    model_type = detect_model_type(config)
    expert_re = MODEL_EXPERT_RES[model_type]
    print(f"Detected model_type={model_type!r}, using expert pattern for {model_type}")

    # Load safetensors headers once per file
    files = sorted(set(weight_map.values()))
    headers: dict[str, dict] = {}
    data_starts: dict[str, int] = {}
    for fname in files:
        h, ds = read_safetensors_header(model_path / fname)
        headers[fname] = h
        data_starts[fname] = ds

    expert_reads: dict[str, dict[str, dict]] = defaultdict(dict)
    totals = {"expert_bytes": 0, "vision_bytes": 0, "non_expert_bytes": 0, "all_bytes": 0}
    by_file: dict[str, dict[str, int]] = {
        "expert": defaultdict(int),
        "non_expert": defaultdict(int),
        "vision": defaultdict(int),
    }

    for name, fname in weight_map.items():
        meta = headers[fname][name]
        nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
        totals["all_bytes"] += nbytes

        m = expert_re.match(name)
        if m:
            totals["expert_bytes"] += nbytes
            by_file["expert"][fname] += nbytes

            layer = m.group("layer")
            component = f"{m.group('proj')}.{m.group('kind')}"
            shape = meta["shape"]
            if not shape or shape[0] <= 0:
                raise ValueError(f"Expert tensor has invalid shape: {name} {shape}")
            num_experts = shape[0]
            expert_reads[layer][component] = {
                "file": fname,
                "abs_offset": data_starts[fname] + meta["data_offsets"][0],
                "expert_stride": nbytes // num_experts,
                "expert_size": nbytes // num_experts,
                "total_size": nbytes,
                "shape": shape,
                "dtype": meta["dtype"],
            }
        elif VISION_RE.match(name):
            totals["vision_bytes"] += nbytes
            by_file["vision"][fname] += nbytes
        else:
            totals["non_expert_bytes"] += nbytes
            by_file["non_expert"][fname] += nbytes

    # Config key names differ: Qwen uses text_config sub-dict; Nemotron is flat
    text_config = config.get("text_config", config)
    num_layers = text_config.get("num_hidden_layers")
    top_k = text_config.get("num_experts_per_tok")
    # Nemotron uses n_routed_experts; Qwen uses num_experts
    num_experts_total = text_config.get("n_routed_experts") or text_config.get("num_experts")

    # Validate uniform per-layer expert size and estimate traffic
    per_layer_expert_bytes = {
        layer: sum(info["expert_size"] for info in comps.values())
        for layer, comps in expert_reads.items()
    }
    moe_layers = len(expert_reads)
    routed_bytes_per_token = None
    if moe_layers and top_k is not None:
        sizes = sorted(set(per_layer_expert_bytes.values()))
        if len(sizes) == 1:
            routed_bytes_per_token = sizes[0] * moe_layers * int(top_k)
        else:
            print(f"  WARNING: non-uniform expert sizes across layers: {sizes[:5]}…")

    return {
        "model_path": str(model_path),
        "model_type": model_type,
        "config": {
            "model_type": config.get("model_type"),
            "hidden_size": text_config.get("hidden_size"),
            "num_hidden_layers": num_layers,
            "num_moe_layers": moe_layers,
            "num_experts": num_experts_total,
            "num_experts_per_tok": top_k,
            "moe_intermediate_size": text_config.get("moe_intermediate_size"),
            "num_attention_heads": text_config.get("num_attention_heads"),
            "num_key_value_heads": text_config.get("num_key_value_heads"),
        },
        "totals": totals,
        "bytes": {
            "per_layer_expert_bytes": per_layer_expert_bytes,
            "routed_bytes_per_token": routed_bytes_per_token,
        },
        "by_file": {k: dict(v) for k, v in by_file.items()},
        "expert_reads": dict(expert_reads),
    }


def print_summary(index: dict, disk_gbps: float | None) -> None:
    cfg = index["config"]
    totals = index["totals"]
    rbt = index["bytes"]["routed_bytes_per_token"]

    print(f"\nModel: {index['model_path']}")
    print(
        f"  type={index['model_type']}  layers={cfg['num_hidden_layers']}"
        f"  moe_layers={cfg['num_moe_layers']}"
        f"  experts={cfg['num_experts']}  top_k={cfg['num_experts_per_tok']}"
        f"  hidden={cfg['hidden_size']}"
    )
    print(
        f"  total={totals['all_bytes']/1024**3:.2f} GiB"
        f"  expert={totals['expert_bytes']/1024**3:.2f} GiB"
        f"  non-expert={totals['non_expert_bytes']/1024**3:.2f} GiB"
    )
    if rbt:
        print(f"  routed per token: {rbt/1024**2:.2f} MiB")
        if disk_gbps:
            for tps in (5, 8, 12, 20):
                need = rbt * tps / 1024**3
                print(f"    {tps:>2} tok/s → {need:.2f} GiB/s  ({need/disk_gbps:.1f}× {disk_gbps:.1f} GiB/s disk)")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    index = build_index(model_path)

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(index, f, indent=2)

    print_summary(index, args.disk_gbps)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
