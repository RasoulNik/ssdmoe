#!/usr/bin/env python3
"""Train and evaluate co-occurrence expert mapper from collected selection data.

Loads per-topic .npz files for both 35B and 122B, trains a per-layer
co-occurrence matrix on the train split, evaluates on the held-out test split,
and produces a learning curve (accuracy vs training tokens).

Two mapper variants are evaluated:
  global  — accumulates all training history equally
  window  — sliding window of last W tokens only (context-aware)

The window variant tests the hypothesis that the 35B→122B expert mapping is
context-dependent: in a physics passage the co-occurrence pattern may differ
from a history passage, so recent history is a better predictor than the global
average.  Window sizes W=50 and W=100 are evaluated.

Structural layer alignment (see experiment doc):
  35B layers 0–39 map 1:1 to 122B layers 0–39.
  122B layers 40–47 have no 35B counterpart.

Usage:
  poetry run python benchmarks/train_expert_mapper.py \\
    --data-dir .run/expert-data \\
    --output   benchmarks/results/expert-mapper.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_EXPERTS = 256
CHECKPOINTS = [50, 100, 200, 500, 1000, 2000, 4000, 8000]


WINDOW_SIZES = [50, 100]  # sliding window sizes to evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=".run/expert-data")
    p.add_argument("--output", default="benchmarks/results/expert-mapper.json")
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="Subset of layers to report (default: every 4th)")
    p.add_argument("--window-sizes", type=int, nargs="*", default=WINDOW_SIZES,
                   help="Sliding window sizes for context-aware mapper")
    return p.parse_args()


class CoOccurrenceMapper:
    """W[i,j] = times 35B expert-i co-activated with 122B expert-j at same token."""

    def __init__(self, layer: int, k: int = 8):
        self.layer = layer
        self.k = k
        self.W = np.zeros((N_EXPERTS, N_EXPERTS), dtype=np.float64)
        self.n = 0

    def update_batch(self, s_small: np.ndarray, s_large: np.ndarray) -> None:
        """s_small, s_large: (n_tok, K) int arrays."""
        # Vectorised outer-product accumulation
        for s, lg in zip(s_small, s_large):
            for i in s:
                for j in lg:
                    self.W[i, j] += 1.0
        self.n += len(s_small)

    def predict(self, s_small: np.ndarray) -> np.ndarray:
        """s_small: (K,) int array.  Returns top-K predicted 122B expert indices."""
        scores = self.W[s_small].sum(axis=0)
        # Normalise each row to avoid bias toward high-frequency 35B experts
        row_sums = self.W[s_small].sum(axis=1, keepdims=True)
        mask = row_sums.squeeze() > 0
        if mask.any():
            normed = np.where(row_sums > 0, self.W[s_small] / (row_sums + 1e-12), 0)
            scores = normed.sum(axis=0)
        return np.argpartition(scores, -self.k)[-self.k:]

    def hit_rate_batch(self, s_small: np.ndarray, s_large: np.ndarray) -> np.ndarray:
        """Returns per-token hit rates: |predicted ∩ actual| / |actual|."""
        hits = []
        for s, lg in zip(s_small, s_large):
            if len(lg) == 0:
                continue
            pred = set(self.predict(s).tolist())
            hits.append(len(pred & set(lg.tolist())) / len(lg))
        return np.array(hits, dtype=np.float32)

    def sparsity(self) -> float:
        return float(np.count_nonzero(self.W)) / self.W.size


class SlidingWindowMapper:
    """Context-aware mapper using only the last W tokens of co-occurrence.

    Instead of accumulating all history, maintains a circular buffer of the
    last W (35B, 122B) pairs and rebuilds W on each prediction from that window.

    At inference time this means: to predict the next token's 122B experts,
    use only the mapping seen in the last W tokens — which reflects the current
    topic/context rather than the global average.
    """

    def __init__(self, layer: int, window: int, k: int = 8):
        self.layer = layer
        self.window = window
        self.k = k
        # Circular buffers: (W, K) int32
        self._buf_s = np.zeros((window, k), dtype=np.int32)
        self._buf_l = np.zeros((window, k), dtype=np.int32)
        self._head = 0      # next write position
        self._filled = 0    # number of valid entries

    def update(self, s: np.ndarray, lg: np.ndarray) -> None:
        """Add one token pair (K,) each."""
        self._buf_s[self._head] = s
        self._buf_l[self._head] = lg
        self._head = (self._head + 1) % self.window
        self._filled = min(self._filled + 1, self.window)

    def _build_W(self) -> np.ndarray:
        if self._filled == 0:
            return np.zeros((N_EXPERTS, N_EXPERTS), dtype=np.float32)
        n = self._filled
        W = np.zeros((N_EXPERTS, N_EXPERTS), dtype=np.float32)
        # Indices of valid entries (oldest first)
        if self._filled < self.window:
            indices = range(self._filled)
        else:
            indices = [(self._head + i) % self.window for i in range(self.window)]
        for idx in indices:
            for i in self._buf_s[idx]:
                for j in self._buf_l[idx]:
                    W[i, j] += 1.0
        return W

    def predict(self, s: np.ndarray) -> np.ndarray:
        W = self._build_W()
        scores = W[s].sum(axis=0)
        row_sums = W[s].sum(axis=1, keepdims=True)
        if (row_sums > 0).any():
            normed = np.where(row_sums > 0, W[s] / (row_sums + 1e-12), 0)
            scores = normed.sum(axis=0)
        return np.argpartition(scores, -self.k)[-self.k:]

    def hit_rate(self, s: np.ndarray, lg: np.ndarray) -> float:
        if len(lg) == 0: return 0.0
        pred = set(self.predict(s).tolist())
        return len(pred & set(lg.tolist())) / len(lg)


def load_topic_data(data_dir: Path, tag: str, topic_id: int, topic_name: str,
                    layers: list[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], int, int]:
    """Returns (train_data, test_data) dicts: layer → (n_tok, K) arrays."""
    path = data_dir / tag / f"topic_{topic_id:02d}_{topic_name}.npz"
    if not path.exists():
        return {}, {}, 0, 0
    npz = np.load(path)
    train_n = int(npz["meta_train_n"][0])
    test_n = int(npz["meta_test_n"][0])
    train_data, test_data = {}, {}
    for l in layers:
        key = f"layer_{l}"
        if key not in npz:
            continue
        arr = npz[key].astype(np.int32)    # (n_tok, K)
        train_data[l] = arr[:train_n]
        test_data[l] = arr[train_n:train_n + test_n]
    return train_data, test_data, train_n, test_n


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    topics_json = data_dir / "topics.json"
    if not topics_json.exists():
        print(f"No topics.json found at {topics_json}"); return
    with open(topics_json) as f:
        topics = json.load(f)

    # Discover available aligned layer pairs from first topic
    first_35b = sorted((data_dir / "35b").glob("topic_00_*.npz"))
    first_122b = sorted((data_dir / "122b").glob("topic_00_*.npz"))
    if not first_35b or not first_122b:
        print("Missing topic_00 files for one or both models."); return

    npz_35b = np.load(first_35b[0])
    npz_122b = np.load(first_122b[0])
    layers_35b = sorted(int(k.split("_")[1]) for k in npz_35b.files if k.startswith("layer_"))
    layers_122b = sorted(int(k.split("_")[1]) for k in npz_122b.files if k.startswith("layer_"))
    k = int(npz_35b["meta_n_tokens"])  # just to get K from shape
    sample_arr = npz_35b[f"layer_{layers_35b[0]}"]
    top_k = sample_arr.shape[1]

    # Aligned layer pairs: identity mapping for shared layers
    shared_layers = sorted(set(layers_35b) & set(layers_122b))
    unmatched_122b = sorted(set(layers_122b) - set(shared_layers))
    print(f"Aligned layer pairs: {len(shared_layers)}  (layers {shared_layers[0]}–{shared_layers[-1]})")
    print(f"Unmatched 122B tail: {len(unmatched_122b)} layers  ({unmatched_122b[0] if unmatched_122b else '—'}–{unmatched_122b[-1] if unmatched_122b else '—'})")
    print(f"Top-K: {top_k},  N_experts: {N_EXPERTS}")
    print(f"Random baseline: {top_k/N_EXPERTS*100:.1f}%")

    # Layers to report in detail
    if args.layers:
        report_layers = [l for l in args.layers if l in shared_layers]
    else:
        report_layers = shared_layers[::4]   # every 4th

    # ── Aggregate all train/test data per layer ──
    print(f"\nLoading data for {len(topics)} topics …", flush=True)
    all_train: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {l: [] for l in shared_layers}
    all_test:  dict[int, list[tuple[np.ndarray, np.ndarray]]] = {l: [] for l in shared_layers}

    for topic in topics:
        tid, name = topic["id"], topic["name"]
        tr_35, te_35, tr_n, te_n = load_topic_data(data_dir, "35b", tid, name, shared_layers)
        tr_12, te_12, _,    _    = load_topic_data(data_dir, "122b", tid, name, shared_layers)
        if not tr_35 or not tr_12:
            print(f"  [skip] topic {tid:02d} {name} — missing data", flush=True)
            continue
        for l in shared_layers:
            if l in tr_35 and l in tr_12:
                n = min(len(tr_35[l]), len(tr_12[l]))
                all_train[l].append((tr_35[l][:n], tr_12[l][:n]))
            if l in te_35 and l in te_12:
                n = min(len(te_35[l]), len(te_12[l]))
                all_test[l].append((te_35[l][:n], te_12[l][:n]))
        print(f"  topic {tid:02d} {name:25s}: train={tr_n}, test={te_n}", flush=True)

    # Concatenate across topics
    train_concat: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    test_concat:  dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for l in shared_layers:
        if all_train[l]:
            s = np.concatenate([p[0] for p in all_train[l]])
            lg = np.concatenate([p[1] for p in all_train[l]])
            train_concat[l] = (s, lg)
        if all_test[l]:
            s = np.concatenate([p[0] for p in all_test[l]])
            lg = np.concatenate([p[1] for p in all_test[l]])
            test_concat[l] = (s, lg)

    total_train = len(next(iter(train_concat.values()))[0]) if train_concat else 0
    total_test  = len(next(iter(test_concat.values()))[0])  if test_concat  else 0
    print(f"\nTotal: {total_train} train tokens, {total_test} test tokens", flush=True)

    # ── Train mappers and evaluate at checkpoints ──
    mappers: dict[int, CoOccurrenceMapper] = {l: CoOccurrenceMapper(l, top_k) for l in shared_layers}
    # Window mappers: trained token-by-token on train set, evaluated on test set
    # (test set evaluation uses window state at the END of training — simulates
    #  online inference where recent context is from the current conversation)
    window_mappers: dict[int, dict[int, SlidingWindowMapper]] = {
        W: {l: SlidingWindowMapper(l, W, top_k) for l in shared_layers}
        for W in args.window_sizes
    }
    checkpoints = [c for c in CHECKPOINTS if c <= total_train]
    if total_train not in checkpoints:
        checkpoints.append(total_train)

    # Print header
    layer_cols = "  ".join(f"L{l:02d}" for l in report_layers)
    print(f"\n{'tokens':>8}  {layer_cols}  ← test hit%  (random={top_k/N_EXPERTS*100:.1f}%)")
    print("─" * (10 + 7 * len(report_layers)))

    checkpoint_results: list[dict] = []
    trained = 0
    cp_idx = 0

    while trained < total_train and cp_idx < len(checkpoints):
        target = checkpoints[cp_idx]
        batch_end = min(target, total_train)
        batch_size = batch_end - trained

        # Train all layers on next batch
        for l in shared_layers:
            if l not in train_concat: continue
            s, lg = train_concat[l]
            mappers[l].update_batch(s[trained:batch_end], lg[trained:batch_end])
            # Update window mappers token by token
            for t in range(trained, batch_end):
                for W in args.window_sizes:
                    window_mappers[W][l].update(s[t], lg[t])

        trained = batch_end

        # Evaluate on full test set
        row = {"token_count": trained, "layers": {}}
        parts = [f"{trained:>8}"]
        for l in shared_layers:
            if l not in test_concat: continue
            s_t, lg_t = test_concat[l]
            hits = mappers[l].hit_rate_batch(s_t, lg_t)
            mean_hit = float(np.mean(hits)) if len(hits) else 0.0
            row["layers"][l] = {
                "test_hit_rate": mean_hit,
                "W_sparsity": mappers[l].sparsity(),
                "n_trained": mappers[l].n,
            }
            if l in report_layers:
                parts.append(f"{mean_hit*100:>6.1f}%")

        print("  ".join(parts), flush=True)
        checkpoint_results.append(row)
        cp_idx += 1

    # ── Final per-layer summary: global vs window mappers ──
    print(f"\n{'═'*75}")
    print(f"Final test hit rate per layer  ({total_train} train tokens, {total_test} test tokens)")
    win_headers = "  ".join(f"W={w:>3}" for w in args.window_sizes)
    print(f"{'Layer':>6}  {'Global':>8}  {win_headers}  {'W_nz%':>7}")
    print("─" * 52)

    window_test_hits: dict[int, dict[int, float]] = {W: {} for W in args.window_sizes}

    for l in shared_layers:
        if l not in test_concat: continue
        s_t, lg_t = test_concat[l]
        global_hits = mappers[l].hit_rate_batch(s_t, lg_t)
        global_mean = float(np.mean(global_hits)) if len(global_hits) else 0.0

        win_means = []
        for W in args.window_sizes:
            wm = window_mappers[W][l]
            w_hits = [wm.hit_rate(s_t[t], lg_t[t]) for t in range(len(s_t))]
            wm_mean = float(np.mean(w_hits)) if w_hits else 0.0
            window_test_hits[W][l] = wm_mean
            win_means.append(f"{wm_mean*100:>7.1f}%")

        win_str = "  ".join(win_means)
        print(f"{l:>6}  {global_mean*100:>7.1f}%  {win_str}  {mappers[l].sparsity()*100:>6.1f}%")

    print(f"\n  Random baseline: {top_k/N_EXPERTS*100:.1f}%")
    print(f"  Unmatched 122B layers (no 35B): {unmatched_122b}")

    # ── Save results ──
    results = {
        "data_dir": str(data_dir),
        "n_topics": len(topics),
        "total_train_tokens": total_train,
        "total_test_tokens": total_test,
        "top_k": top_k,
        "n_experts": N_EXPERTS,
        "random_baseline": top_k / N_EXPERTS,
        "aligned_layers": shared_layers,
        "unmatched_122b_layers": unmatched_122b,
        "window_sizes": args.window_sizes,
        "checkpoints": checkpoint_results,
        "final_per_layer": {
            str(l): {
                "global_test_hit_rate": checkpoint_results[-1]["layers"].get(l, {}).get("test_hit_rate", 0),
                "window_test_hit_rates": {str(W): window_test_hits[W].get(l, 0) for W in args.window_sizes},
                "W_sparsity": mappers[l].sparsity(),
            }
            for l in shared_layers if l in test_concat
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out}", flush=True)


if __name__ == "__main__":
    main()
