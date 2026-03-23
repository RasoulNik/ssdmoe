from __future__ import annotations

import json
import os
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np


EXPERT_RE = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.mlp\.switch_mlp\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight|scales|biases)$"
)
VISION_RE = re.compile(r"^(vision_tower|model\.visual|visual)")
DTYPE_TO_NP = {
    "F32": np.float32,
    "U32": np.uint32,
    "I32": np.int32,
    "U16": np.uint16,
    "I16": np.int16,
    "U8": np.uint8,
    "I8": np.int8,
    "BOOL": np.bool_,
}


def load_config(model_path: Path) -> dict:
    with (model_path / "config.json").open() as f:
        return json.load(f)


def load_weight_map(model_path: Path) -> dict[str, str]:
    with (model_path / "model.safetensors.index.json").open() as f:
        return json.load(f)["weight_map"]


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def classify_tensor(name: str) -> str:
    if EXPERT_RE.match(name):
        return "expert"
    if VISION_RE.match(name):
        return "vision"
    return "non_expert"


def list_non_expert_text_tensors(model_path: Path) -> list[str]:
    weight_map = load_weight_map(model_path)
    return sorted(name for name in weight_map if classify_tensor(name) == "non_expert")


def list_expert_tensors(model_path: Path) -> list[str]:
    weight_map = load_weight_map(model_path)
    return sorted(name for name in weight_map if classify_tensor(name) == "expert")


def list_expert_aux_tensors(model_path: Path) -> list[str]:
    return [name for name in list_expert_tensors(model_path) if not name.endswith(".weight")]


def group_expert_tensor_map(tensors: dict[str, mx.array]) -> dict[int, dict[str, mx.array]]:
    grouped: dict[int, dict[str, mx.array]] = defaultdict(dict)
    for name, tensor in tensors.items():
        match = EXPERT_RE.match(name)
        if match is None:
            continue
        layer_idx = int(match.group("layer"))
        component = f"{match.group('proj')}.{match.group('kind')}"
        grouped[layer_idx][component] = tensor
    return dict(grouped)


def group_weight_names_by_file(
    model_path: Path, names: Iterable[str]
) -> dict[str, list[str]]:
    weight_map = load_weight_map(model_path)
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in names:
        grouped[weight_map[name]].append(name)
    return grouped


def _tensor_to_mx(data: bytes, dtype_name: str, shape: list[int]) -> mx.array:
    if dtype_name == "BF16":
        raw = np.frombuffer(data, dtype=np.uint16).reshape(shape)
        return mx.view(mx.asarray(raw), mx.bfloat16)

    if dtype_name not in DTYPE_TO_NP:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    raw = np.frombuffer(data, dtype=DTYPE_TO_NP[dtype_name]).reshape(shape)
    return mx.asarray(raw)


def load_named_tensors(model_path: Path, names: Iterable[str]) -> dict[str, mx.array]:
    grouped = group_weight_names_by_file(model_path, names)
    out: dict[str, mx.array] = {}

    for file_name, file_names in grouped.items():
        shard_path = model_path / file_name
        header, data_start = read_safetensors_header(shard_path)
        fd = os.open(shard_path, os.O_RDONLY)
        try:
            for name in file_names:
                meta = header[name]
                start, end = meta["data_offsets"]
                data = os.pread(fd, end - start, data_start + start)
                out[name] = _tensor_to_mx(data, meta["dtype"], meta["shape"])
        finally:
            os.close(fd)

    return out


def load_non_expert_text_weights(model_path: Path) -> dict[str, mx.array]:
    return load_named_tensors(model_path, list_non_expert_text_tensors(model_path))


def load_expert_aux_weights(model_path: Path) -> dict[int, dict[str, mx.array]]:
    return group_expert_tensor_map(load_named_tensors(model_path, list_expert_aux_tensors(model_path)))
