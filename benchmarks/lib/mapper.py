"""Co-occurrence expert mappers for cross-model expert index prediction.

Two variants:
  CoOccurrenceMapper    — accumulates all history in a global weight matrix W[i,j]
  SlidingWindowMapper   — context-aware; uses only the last `window` tokens

Both are pure-numpy with no model dependency.  `n_experts` is read from the
index config at construction time (via lib.index.load_index_config) rather
than being hardcoded.

Usage:
    from lib.mapper import CoOccurrenceMapper, SlidingWindowMapper
    from lib.index import load_index_config

    cfg = load_index_config(Path(args.index))
    mapper = CoOccurrenceMapper(layer=0, n_experts=cfg.n_experts, k=cfg.n_experts_per_tok)
    mapper.update_batch(s_small, s_large)   # (n_tok, K) int32 arrays
    hits = mapper.hit_rate_batch(s_test_small, s_test_large)
"""
from __future__ import annotations

import numpy as np


class CoOccurrenceMapper:
    """W[i,j] = number of times small-model expert-i co-activated with large-model expert-j.

    Accumulates all training history equally.

    Args:
        layer:     MoE layer index (used for identification only, not computation).
        n_experts: Total number of experts in the model (from IndexConfig.n_experts).
        k:         Top-K for prediction (same as routed top-k used during collection).
    """

    def __init__(self, layer: int, n_experts: int, k: int = 8) -> None:
        self.layer = layer
        self.n_experts = n_experts
        self.k = k
        self.W = np.zeros((n_experts, n_experts), dtype=np.float64)
        self.n = 0

    def update_batch(self, s_small: np.ndarray, s_large: np.ndarray) -> None:
        """Accumulate co-occurrence for a batch of token pairs.

        Args:
            s_small: (n_tok, K) int array — small-model expert selections.
            s_large: (n_tok, K) int array — large-model expert selections.
        """
        for s, lg in zip(s_small, s_large):
            for i in s:
                for j in lg:
                    self.W[i, j] += 1.0
        self.n += len(s_small)

    def predict(self, s_small: np.ndarray) -> np.ndarray:
        """Predict top-K large-model expert indices from small-model selection.

        Args:
            s_small: (K,) int array of small-model expert indices.

        Returns:
            (K,) int array of predicted large-model expert indices.
        """
        row_sums = self.W[s_small].sum(axis=1, keepdims=True)
        if (row_sums.squeeze() > 0).any():
            normed = np.where(row_sums > 0, self.W[s_small] / (row_sums + 1e-12), 0)
            scores = normed.sum(axis=0)
        else:
            scores = self.W[s_small].sum(axis=0)
        return np.argpartition(scores, -self.k)[-self.k:]

    def hit_rate_batch(self, s_small: np.ndarray, s_large: np.ndarray) -> np.ndarray:
        """Per-token hit rates: |predicted ∩ actual| / |actual|.

        Args:
            s_small: (n_tok, K) int array.
            s_large: (n_tok, K) int array.

        Returns:
            (n_tok,) float32 array of hit rates in [0, 1].
        """
        hits = []
        for s, lg in zip(s_small, s_large):
            if len(lg) == 0:
                continue
            pred = set(self.predict(s).tolist())
            hits.append(len(pred & set(lg.tolist())) / len(lg))
        return np.array(hits, dtype=np.float32)

    def sparsity(self) -> float:
        """Fraction of non-zero entries in W (0 = all-zero, 1 = fully dense)."""
        return float(np.count_nonzero(self.W)) / self.W.size


class SlidingWindowMapper:
    """Context-aware mapper; uses only the last `window` tokens of co-occurrence.

    Maintains a circular buffer of the last W (small, large) token pairs and
    rebuilds the weight matrix on each prediction.  Captures topic/context
    shifts that the global CoOccurrenceMapper averages away.

    Args:
        layer:     MoE layer index.
        window:    Number of recent tokens to keep in the circular buffer.
        n_experts: Total number of experts in the model.
        k:         Top-K for prediction.
    """

    def __init__(self, layer: int, window: int, n_experts: int, k: int = 8) -> None:
        self.layer = layer
        self.window = window
        self.n_experts = n_experts
        self.k = k
        self._buf_s = np.zeros((window, k), dtype=np.int32)
        self._buf_l = np.zeros((window, k), dtype=np.int32)
        self._head = 0
        self._filled = 0

    def update(self, s: np.ndarray, lg: np.ndarray) -> None:
        """Add one token pair ((K,) arrays each) to the circular buffer."""
        self._buf_s[self._head] = s
        self._buf_l[self._head] = lg
        self._head = (self._head + 1) % self.window
        self._filled = min(self._filled + 1, self.window)

    def _build_W(self) -> np.ndarray:
        if self._filled == 0:
            return np.zeros((self.n_experts, self.n_experts), dtype=np.float32)
        W = np.zeros((self.n_experts, self.n_experts), dtype=np.float32)
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
        """Predict top-K large-model expert indices from small-model selection."""
        W = self._build_W()
        row_sums = W[s].sum(axis=1, keepdims=True)
        if (row_sums > 0).any():
            normed = np.where(row_sums > 0, W[s] / (row_sums + 1e-12), 0)
            scores = normed.sum(axis=0)
        else:
            scores = W[s].sum(axis=0)
        return np.argpartition(scores, -self.k)[-self.k:]

    def hit_rate(self, s: np.ndarray, lg: np.ndarray) -> float:
        """Hit rate for one token pair: |predicted ∩ actual| / |actual|."""
        if len(lg) == 0:
            return 0.0
        pred = set(self.predict(s).tolist())
        return len(pred & set(lg.tolist())) / len(lg)
