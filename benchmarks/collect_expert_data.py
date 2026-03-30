#!/usr/bin/env python3
"""Collect per-token expert selections from ALL MoE layers via prefill.

Strategy: feed each topic as a single forward pass (prefill).  The MoE switch
fires once per layer with indices shape (seq_len, K), giving us token-level
selections without any generation loop.

Data layout
-----------
  .run/expert-data/{model_tag}/topic_{i:02d}_{name}.npz
    layer_0  : int16 (n_tok, K)   ← sorted selected expert indices
    layer_1  : int16 (n_tok, K)
    ...
    layer_N  : int16 (n_tok, K)
    meta_n_tokens    : scalar
    meta_train_n     : scalar   (first 80% = train)
    meta_test_n      : scalar   (last  20% = test)

  .run/expert-data/topics.json
    [{id, name, 35b_tokens, 122b_tokens}, ...]

Usage
-----
  poetry run python benchmarks/collect_expert_data.py \\
    --model  ~/.cache/.../Qwen3.5-35B-A3B-4bit/snapshots/... \\
    --index  .run/qwen35-35b-expert-index.json \\
    --tag    35b \\
    --out-dir .run/expert-data

  poetry run python benchmarks/collect_expert_data.py \\
    --model  ~/.cache/.../Qwen3.5-122B-A10B-4bit/snapshots/... \\
    --index  .run/qwen35-122b-expert-index.json \\
    --tag    122b \\
    --out-dir .run/expert-data
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.loader import ensure_src_path
ensure_src_path()

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--native-reader", required=True)
    p.add_argument("--tag", required=True, help="e.g. 35b or 122b")
    p.add_argument("--out-dir", default=".run/expert-data")
    p.add_argument("--max-tokens", type=int, default=1000,
                   help="Max tokens per topic (default 1000)")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--component-workers", type=int, default=3)
    p.add_argument("--topics", type=int, nargs="*", default=None,
                   help="Topic indices to collect (default: all)")
    p.add_argument("--topics-file",
                   default=str(Path(__file__).resolve().parent / "data" / "topics.jsonl"),
                   help="JSONL file with topic data (default: benchmarks/data/topics.jsonl)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    from streaming_moe.runtime import build_streamed_model, set_routed_top_k, _get_moe_module
    from streaming_moe.streamed_switch import StreamedSwitchGLU
    from streaming_moe.prefetch_switch import PrefetchingStreamedSwitchGLU
    from mlx_lm.models.cache import make_prompt_cache

    top_k = args.top_k
    with open(args.topics_file) as _f:
        _all_topics = [json.loads(line) for line in _f if line.strip()]
    TOPICS = [(t["name"], t["text"]) for t in _all_topics]
    topic_indices = args.topics if args.topics is not None else list(range(len(TOPICS)))

    print(f"Loading model [{args.tag}] …", flush=True)
    model, tokenizer, _, _ = build_streamed_model(
        Path(args.model), Path(args.index),
        native_reader_path=Path(args.native_reader),
        component_workers=args.component_workers,
    )
    set_routed_top_k(model, top_k)

    # Discover all MoE layer indices
    tm = getattr(getattr(model, "language_model", model), "model", model)
    all_moe_layers: list[int] = []
    for i, layer in enumerate(tm.layers):
        moe, _ = _get_moe_module(layer)
        if moe is not None and getattr(moe, "switch_mlp", None) is not None:
            all_moe_layers.append(i)
    print(f"MoE layers: {len(all_moe_layers)}  ({all_moe_layers[0]}–{all_moe_layers[-1]})", flush=True)

    # ── Recording hook (handles both single-token and multi-token) ──
    selections: dict[int, list[np.ndarray]] = {}  # layer → list of (K,) arrays

    class _Hook(nn.Module):
        def __init__(self, inner, layer_idx: int):
            super().__init__()
            self._inner = inner
            self._l = layer_idx

        def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
            idx = np.array(indices.tolist(), dtype=np.int16)
            idx = idx.reshape(-1, top_k)        # (n_tok, K) in all cases
            for row in idx:
                selections[self._l].append(row.copy())
            return self._inner(x, indices)

    # Install hooks on all MoE layers
    for i, layer in enumerate(tm.layers):
        if i not in all_moe_layers:
            continue
        moe, _ = _get_moe_module(layer)
        sw = getattr(moe, "switch_mlp", None)
        if sw and isinstance(sw, (StreamedSwitchGLU, PrefetchingStreamedSwitchGLU)):
            moe.switch_mlp = _Hook(sw, i)

    topic_meta = []

    for topic_idx in topic_indices:
        name, text = TOPICS[topic_idx]
        out_path = out_dir / f"topic_{topic_idx:02d}_{name}.npz"

        if out_path.exists():
            print(f"  [skip] topic {topic_idx:02d} {name} already exists", flush=True)
            existing = np.load(out_path)
            n = int(existing["meta_n_tokens"])
            topic_meta.append({"id": topic_idx, "name": name, f"{args.tag}_tokens": n,
                                "train_n": int(existing["meta_train_n"]),
                                "test_n": int(existing["meta_test_n"])})
            continue

        # Tokenize (strip leading whitespace, cap at max_tokens)
        tokens = tokenizer.encode(text.strip())[:args.max_tokens]
        n_tok = len(tokens)
        print(f"  Topic {topic_idx:02d} [{name}]: {n_tok} tokens … ", end="", flush=True)

        # Clear buffers
        for l in all_moe_layers:
            selections[l] = []

        # Single prefill forward pass
        t0 = time.perf_counter()
        arr = mx.array(tokens, dtype=mx.uint32)
        cache = make_prompt_cache(model)
        out = model(arr[None], cache=cache)
        mx.eval(out)
        elapsed = time.perf_counter() - t0

        # Verify capture counts
        n_captured = len(selections[all_moe_layers[0]])
        print(f"{n_captured} tok captured  ({n_captured/elapsed:.0f} tok/s)", flush=True)

        # Build save dict: layer_L → (n_tok, K) int16
        train_n = int(n_captured * 0.8)
        test_n = n_captured - train_n
        save_dict: dict[str, np.ndarray] = {}
        for l in all_moe_layers:
            sel = selections[l]
            arr_l = np.stack(sel[:n_captured], axis=0).astype(np.int16)  # (n_tok, K)
            save_dict[f"layer_{l}"] = arr_l
        save_dict["meta_n_tokens"] = np.array([n_captured], dtype=np.int32)
        save_dict["meta_train_n"] = np.array([train_n], dtype=np.int32)
        save_dict["meta_test_n"] = np.array([test_n], dtype=np.int32)

        np.savez_compressed(out_path, **save_dict)
        print(f"    → saved {out_path}  (train={train_n}, test={test_n})", flush=True)

        topic_meta.append({"id": topic_idx, "name": name, f"{args.tag}_tokens": n_captured,
                            "train_n": train_n, "test_n": test_n})

    # Update topics.json
    topics_json = Path(args.out_dir) / "topics.json"
    existing_meta: list[dict] = []
    if topics_json.exists():
        with open(topics_json) as f:
            existing_meta = json.load(f)
    existing_by_id = {t["id"]: t for t in existing_meta}
    for m in topic_meta:
        tid = m["id"]
        if tid in existing_by_id:
            existing_by_id[tid].update(m)
        else:
            existing_by_id[tid] = m
    with open(topics_json, "w") as f:
        json.dump(sorted(existing_by_id.values(), key=lambda x: x["id"]), f, indent=2)

    print(f"\nDone. topics.json updated → {topics_json}", flush=True)


if __name__ == "__main__":
    main()
