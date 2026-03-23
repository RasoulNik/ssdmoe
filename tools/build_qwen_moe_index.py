#!/usr/bin/env python3
"""Build a routed-expert read index for Qwen 3.5 MoE MLX checkpoints.

This is designed for low-disk setups where we cannot afford to repack experts
into a second on-disk copy. Instead, we stream expert slices directly from the
original safetensors shards using byte offsets.

The script currently targets the MLX-converted Qwen 3.5 35B A3B layout:

  language_model.model.layers.{layer}.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}

It also emits high-level size accounting so we can reason about RAM, page
cache, and SSD throughput before building the inference path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
from collections import defaultdict
from pathlib import Path


EXPERT_RE = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.mlp\.switch_mlp\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight|scales|biases)$"
)

VISION_RE = re.compile(r"^(vision_tower|model\.visual|visual)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build routed-expert byte-offset index for a local Qwen MoE model"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the local Hugging Face snapshot directory",
    )
    parser.add_argument(
        "--output",
        default="expert_index.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--disk-gbps",
        type=float,
        default=None,
        help="Optional measured disk throughput in GB/s for simple tok/s upper bounds",
    )
    return parser.parse_args()


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def tensor_nbytes(meta: dict) -> int:
    start, end = meta["data_offsets"]
    return end - start


def classify_tensor(name: str) -> str:
    if EXPERT_RE.match(name):
        return "expert"
    if VISION_RE.match(name):
        return "vision"
    return "non_expert"


def build_index(model_path: Path) -> dict:
    index_path = model_path / "model.safetensors.index.json"
    config_path = model_path / "config.json"

    with index_path.open() as f:
        raw_index = json.load(f)
    with config_path.open() as f:
        config = json.load(f)

    weight_map = raw_index["weight_map"]
    files = sorted(set(weight_map.values()))

    headers = {}
    data_starts = {}
    for file_name in files:
        header, data_start = read_safetensors_header(model_path / file_name)
        headers[file_name] = header
        data_starts[file_name] = data_start

    expert_reads: dict[str, dict[str, dict]] = defaultdict(dict)
    totals = {
        "expert_bytes": 0,
        "vision_bytes": 0,
        "non_expert_bytes": 0,
        "all_bytes": 0,
    }
    non_expert_by_file: dict[str, int] = defaultdict(int)
    vision_by_file: dict[str, int] = defaultdict(int)
    expert_by_file: dict[str, int] = defaultdict(int)

    for name, file_name in weight_map.items():
        meta = headers[file_name][name]
        nbytes = tensor_nbytes(meta)
        totals["all_bytes"] += nbytes
        category = classify_tensor(name)

        if category == "expert":
            totals["expert_bytes"] += nbytes
            expert_by_file[file_name] += nbytes

            match = EXPERT_RE.match(name)
            assert match is not None
            layer = match.group("layer")
            component = f"{match.group('proj')}.{match.group('kind')}"
            shape = meta["shape"]
            if not shape:
                raise ValueError(f"Expert tensor has empty shape: {name}")
            num_experts = shape[0]
            if num_experts <= 0:
                raise ValueError(f"Invalid expert dimension for {name}: {shape}")

            expert_reads[layer][component] = {
                "file": file_name,
                "abs_offset": data_starts[file_name] + meta["data_offsets"][0],
                "expert_stride": nbytes // num_experts,
                "expert_size": nbytes // num_experts,
                "total_size": nbytes,
                "shape": shape,
                "dtype": meta["dtype"],
            }
        elif category == "vision":
            totals["vision_bytes"] += nbytes
            vision_by_file[file_name] += nbytes
        else:
            totals["non_expert_bytes"] += nbytes
            non_expert_by_file[file_name] += nbytes

    text_config = config.get("text_config", config)
    num_layers = text_config.get("num_hidden_layers")
    top_k = text_config.get("num_experts_per_tok")

    per_layer_expert_bytes = {}
    for layer, components in expert_reads.items():
        per_layer_expert_bytes[layer] = sum(
            info["expert_size"] for info in components.values()
        )

    routed_bytes_per_token = None
    if num_layers is not None and top_k is not None and per_layer_expert_bytes:
        unique_sizes = sorted(set(per_layer_expert_bytes.values()))
        if len(unique_sizes) != 1:
            raise ValueError(
                f"Expected uniform expert size across layers, got: {unique_sizes}"
            )
        routed_bytes_per_token = unique_sizes[0] * int(num_layers) * int(top_k)

    summary = {
        "model_path": str(model_path),
        "config": {
            "model_type": config.get("model_type"),
            "text_model_type": text_config.get("model_type"),
            "hidden_size": text_config.get("hidden_size"),
            "num_hidden_layers": num_layers,
            "num_experts": text_config.get("num_experts"),
            "num_experts_per_tok": top_k,
            "moe_intermediate_size": text_config.get("moe_intermediate_size"),
            "shared_expert_intermediate_size": text_config.get(
                "shared_expert_intermediate_size"
            ),
            "num_attention_heads": text_config.get("num_attention_heads"),
            "num_key_value_heads": text_config.get("num_key_value_heads"),
        },
        "totals": totals,
        "bytes": {
            "per_layer_expert_bytes": per_layer_expert_bytes,
            "routed_bytes_per_token": routed_bytes_per_token,
        },
        "by_file": {
            "expert": expert_by_file,
            "non_expert": non_expert_by_file,
            "vision": vision_by_file,
        },
        "expert_reads": dict(expert_reads),
    }

    return summary


def print_summary(index: dict, disk_gbps: float | None) -> None:
    cfg = index["config"]
    totals = index["totals"]
    routed_bytes_per_token = index["bytes"]["routed_bytes_per_token"]

    print(f"Model: {index['model_path']}")
    print(
        "Text config:"
        f" layers={cfg['num_hidden_layers']},"
        f" experts={cfg['num_experts']},"
        f" top_k={cfg['num_experts_per_tok']},"
        f" hidden={cfg['hidden_size']}"
    )
    print(
        "Tensor bytes:"
        f" total={totals['all_bytes'] / 1024**3:.2f} GiB,"
        f" non_expert={totals['non_expert_bytes'] / 1024**3:.2f} GiB,"
        f" expert={totals['expert_bytes'] / 1024**3:.2f} GiB,"
        f" vision={totals['vision_bytes'] / 1024**3:.2f} GiB"
    )

    if routed_bytes_per_token is not None:
        per_expert = routed_bytes_per_token // (
            cfg["num_hidden_layers"] * cfg["num_experts_per_tok"]
        )
        print(
            "Routed expert traffic:"
            f" expert={per_expert / 1024**2:.4f} MiB,"
            f" token={routed_bytes_per_token / 1024**3:.4f} GiB"
        )
        if disk_gbps:
            for tps in (5, 10, 20, 30):
                need = routed_bytes_per_token * tps / 1024**3
                print(
                    f"  {tps:>2} tok/s requires about {need:.2f} GiB/s"
                    f" ({need / disk_gbps:.1f}x a {disk_gbps:.2f} GiB/s disk)"
                )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    index = build_index(model_path)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(index, f, indent=2)

    print_summary(index, args.disk_gbps)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
