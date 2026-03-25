#!/usr/bin/env python3
"""Online expert-index mapping: learn to translate 35B expert indices → 122B.

Runs in three modes:
  collect  -- run one model and save per-token expert selections to disk
  train    -- load both saved selections, train co-occurrence mapper, show accuracy curve
  run      -- collect + train in sequence (default)

Usage:
  # Collect 35B selections (fast, ~6 tok/s)
  poetry run python benchmarks/bench_expert_mapper.py collect \
    --model  ~/.cache/.../Qwen3.5-35B-A3B-4bit/snapshots/... \
    --index  .run/qwen35-35b-expert-index.json \
    --out    .run/selections-35b.npz \
    --tokens 500

  # Collect 122B selections (slow, ~1.4 tok/s)
  poetry run python benchmarks/bench_expert_mapper.py collect \
    --model  ~/.cache/.../Qwen3.5-122B-A10B-4bit/snapshots/... \
    --index  .run/qwen35-122b-expert-index.json \
    --out    .run/selections-122b.npz \
    --tokens 500

  # Train mapper on saved selections
  poetry run python benchmarks/bench_expert_mapper.py train \
    --small .run/selections-35b.npz \
    --large .run/selections-122b.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

N_EXPERTS = 256
PROBE_LAYERS = [0, 12, 24, 36]
CHECKPOINTS = [50, 100, 200, 300, 500]

PROMPTS = [
    "The cat sat on the mat.",
    "Explain how neural networks learn from data.",
    "The capital of France is Paris.",
    "Write a short poem about the ocean.",
    "What is the difference between a stack and a queue?",
    "The French Revolution began in 1789.",
    "Describe the water cycle in detail.",
    "Implement a binary search algorithm in Python.",
    "What causes earthquakes?",
    "Tell me about the history of the Roman Empire.",
    "How does photosynthesis work?",
    "Explain the concept of recursion with an example.",
    "What are the main causes of climate change?",
    "Describe the architecture of a transformer model.",
]


# ── Collect mode ───────────────────────────────────────────────────────────────

def collect(args):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.cache import make_prompt_cache
    from streaming_moe.runtime import build_streamed_model, set_routed_top_k
    from streaming_moe.streamed_switch import StreamedSwitchGLU
    from streaming_moe.prefetch_switch import PrefetchingStreamedSwitchGLU

    probe_set = set(args.probe_layers)
    selections: dict[int, list[list[int]]] = defaultdict(list)  # layer → [token_selections]

    class _Hook(nn.Module):
        def __init__(self, inner, layer_idx):
            super().__init__()
            self._inner = inner
            self._l = layer_idx
        def __call__(self, x, indices):
            idx = np.array(indices.tolist(), dtype=np.int16)
            selections[self._l].append(sorted(np.unique(idx).tolist()))
            return self._inner(x, indices)

    print(f"Loading model {Path(args.model).name[:40]} …", flush=True)
    model, tokenizer, _, _ = build_streamed_model(
        Path(args.model), Path(args.index),
        native_reader_path=Path(args.native_reader),
        component_workers=3,
    )
    set_routed_top_k(model, args.top_k)

    # Install hooks only on probe layers
    from streaming_moe.runtime import _get_moe_module
    tm = getattr(getattr(model, "language_model", model), "model", model)
    for i, layer in enumerate(tm.layers):
        if i not in probe_set: continue
        moe, _ = _get_moe_module(layer)
        if moe is None: continue
        sw = getattr(moe, "switch_mlp", None)
        if sw and isinstance(sw, (StreamedSwitchGLU, PrefetchingStreamedSwitchGLU)):
            moe.switch_mlp = _Hook(sw, i)
    print(f"Hooked layers: {sorted(probe_set)}", flush=True)

    token_count = 0
    prompt_idx = 0
    tokens_per_prompt = max(8, args.tokens // len(PROMPTS))

    from mlx_lm.generate import generate_step

    while token_count < args.tokens:
        prompt = PROMPTS[prompt_idx % len(PROMPTS)]
        prompt_idx += 1
        n = min(tokens_per_prompt, args.tokens - token_count)

        before = {l: len(selections[l]) for l in probe_set}

        toks = tokenizer.encode(prompt)
        arr = mx.array(toks, dtype=mx.uint32)
        cache = make_prompt_cache(model)
        if arr.size > 1:
            model(arr[:-1][None], cache=cache)
            mx.eval([c.state for c in cache])
        last = arr[-1:]
        eos = set(getattr(tokenizer, "eos_token_ids", []) or [])
        for token, _ in generate_step(last, model, max_tokens=n, prompt_cache=cache):
            mx.eval(token)
            if int(token) in eos: break

        added = min(len(selections[l]) - before[l] for l in probe_set if l in selections)
        token_count += added
        print(f"  '{prompt[:45]}' → +{added} tokens  (total: {token_count})", flush=True)

    # Save as structured numpy array
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {}
    for l in sorted(probe_set):
        sel = selections[l]
        # Pad each selection to top_k length (some layers might have fewer unique)
        arr = np.zeros((len(sel), args.top_k), dtype=np.int16)
        for t, s in enumerate(sel):
            s_arr = np.array(s[:args.top_k], dtype=np.int16)
            arr[t, :len(s_arr)] = s_arr
        save_dict[f"layer_{l}"] = arr
    save_dict["meta_top_k"] = np.array([args.top_k])
    save_dict["meta_n_experts"] = np.array([N_EXPERTS])
    np.savez(out, **save_dict)
    print(f"\nSaved {token_count} tokens × {len(probe_set)} layers → {out}", flush=True)


# ── Train mode ─────────────────────────────────────────────────────────────────

class CoOccurrenceMapper:
    def __init__(self, n_experts=256, top_k=8):
        self.W = np.zeros((n_experts, n_experts), dtype=np.float32)
        self.n = 0
        self.top_k = top_k

    def update(self, s_small: list[int], s_large: list[int]):
        for i in s_small:
            for j in s_large:
                self.W[i, j] += 1.0
        self.n += 1

    def predict(self, s_small: list[int]) -> set[int]:
        if self.n == 0: return set()
        scores = np.zeros(N_EXPERTS, dtype=np.float32)
        for i in s_small:
            row = self.W[i]
            s = row.sum()
            if s > 0:
                scores += row / s
        return set(np.argpartition(scores, -self.top_k)[-self.top_k:].tolist())

    def hit_rate(self, s_small: list[int], s_large: list[int]) -> float:
        if not s_large: return 0.0
        pred = self.predict(s_small)
        return len(pred & set(s_large)) / len(s_large)

    def w_sparsity(self) -> float:
        return float(np.count_nonzero(self.W)) / self.W.size


def train(args):
    print(f"Loading selections …", flush=True)
    small = np.load(args.small)
    large = np.load(args.large)

    top_k = int(small["meta_top_k"][0])
    layers = [int(k.split("_")[1]) for k in small.files if k.startswith("layer_")]
    layers = sorted(l for l in layers if f"layer_{l}" in large.files)
    print(f"Layers: {layers},  top_k={top_k}", flush=True)

    mappers = {l: CoOccurrenceMapper(N_EXPERTS, top_k) for l in layers}

    n_tokens = min(small[f"layer_{layers[0]}"].shape[0],
                   large[f"layer_{layers[0]}"].shape[0])
    print(f"Token pairs available: {n_tokens}", flush=True)

    # Per-layer token arrays
    s_data = {l: small[f"layer_{l}"] for l in layers}
    l_data = {l: large[f"layer_{l}"] for l in layers}

    # Eval window: last 50 tokens seen so far (leave-one-out style)
    EVAL_WINDOW = 50
    checkpoint_results = []
    next_cp = 0

    print(f"\n{'tokens':>8}  " + "  ".join(f"L{l:02d} hit%" for l in layers) +
          f"  (random={top_k/N_EXPERTS*100:.1f}%)", flush=True)
    print("─" * (8 + 12 * len(layers)), flush=True)

    for t in range(n_tokens):
        # Update all layer mappers with token t
        for l in layers:
            s = s_data[l][t].tolist()
            lg = l_data[l][t].tolist()
            mappers[l].update(s, lg)

        # Checkpoint evaluation
        if next_cp < len(CHECKPOINTS) and (t + 1) >= CHECKPOINTS[next_cp]:
            # Evaluate on last EVAL_WINDOW tokens (already seen = in-sample,
            # but good enough to show whether W is learning signal vs noise)
            start = max(0, t + 1 - EVAL_WINDOW)
            row = {"token_count": t + 1, "layers": {}}
            parts = [f"{t+1:>8}"]
            for l in layers:
                hits = []
                for tt in range(start, t + 1):
                    s = s_data[l][tt].tolist()
                    lg = l_data[l][tt].tolist()
                    hits.append(mappers[l].hit_rate(s, lg))
                mean_hit = float(np.mean(hits)) if hits else 0.0
                row["layers"][l] = {
                    "hit_rate": mean_hit,
                    "n_trained": mappers[l].n,
                    "W_sparsity": mappers[l].w_sparsity(),
                }
                parts.append(f"{mean_hit*100:>9.1f}%")
            print("  ".join(parts), flush=True)
            checkpoint_results.append(row)
            next_cp += 1

        if t + 1 >= (args.tokens or n_tokens):
            break

    # Final summary
    print(f"\n{'═'*60}", flush=True)
    print(f"Random baseline: {top_k/N_EXPERTS*100:.1f}%  (K={top_k}/{N_EXPERTS})", flush=True)
    print(f"W matrix sparsity per layer:", flush=True)
    for l in layers:
        sp = mappers[l].w_sparsity()
        print(f"  Layer {l:>2}: {sp*100:.1f}% of {N_EXPERTS}×{N_EXPERTS} entries nonzero  "
              f"(fully uniform=100%, perfect sparse permutation≈{top_k/N_EXPERTS*100:.1f}%)", flush=True)

    # Save results
    results = {
        "small_file": str(args.small),
        "large_file": str(args.large),
        "top_k": top_k,
        "n_experts": N_EXPERTS,
        "layers": layers,
        "random_baseline": top_k / N_EXPERTS,
        "checkpoints": checkpoint_results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out}", flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("collect")
    c.add_argument("--model", required=True)
    c.add_argument("--index", required=True)
    c.add_argument("--native-reader", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--tokens", type=int, default=500)
    c.add_argument("--top-k", type=int, default=8)
    c.add_argument("--probe-layers", type=int, nargs="+", default=PROBE_LAYERS)

    t = sub.add_parser("train")
    t.add_argument("--small", required=True, help=".npz from 35B collect run")
    t.add_argument("--large", required=True, help=".npz from 122B collect run")
    t.add_argument("--tokens", type=int, default=None, help="limit token pairs")
    t.add_argument("--output", default="benchmarks/results/expert-mapper.json")

    args = p.parse_args()
    if args.cmd == "collect":
        collect(args)
    elif args.cmd == "train":
        train(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
