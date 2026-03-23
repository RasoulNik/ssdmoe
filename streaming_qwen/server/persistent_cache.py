"""Disk-backed prompt KV cache.

Wraps LRUPromptCache with transparent safetensors persistence:
  - Memory budget : max_bytes  (existing LRU eviction)
  - Disk budget   : disk_max_bytes  (default 5 × max_bytes)
  - Checkpoint entries are queued during inference, then flushed synchronously
    after generation completes via flush_pending_saves().  This keeps all MLX /
    Metal operations on the HTTP handler thread — background threads must never
    call mx.savez because mx.savez issues Metal GPU→CPU blits that race with
    active inference on the same Metal command queue.
  - On server startup, load_from_disk() restores saved entries into memory,
    so clients skip cold-start prefill after a restart.

File layout in disk_dir:
  <sha256[:20]>.safetensors   – serialised prompt_cache (MLX safetensors)
  <sha256[:20]>.json          – lightweight sidecar: model, n_tokens, bytes
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from mlx_lm.server import LRUPromptCache
from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache


class PersistentPromptCache:
    """LRUPromptCache with transparent disk persistence for warm restarts."""

    def __init__(
        self,
        *,
        max_size: int = 8,
        max_bytes: int = 1 << 30,        # 1 GB memory
        disk_dir: Path | str | None = None,
        disk_max_bytes: int = 0,          # 0 → 5 × max_bytes
    ) -> None:
        self._mem = LRUPromptCache(max_size=max_size, max_bytes=max_bytes)
        self._disk_dir = Path(disk_dir) if disk_dir else None
        self._disk_max_bytes = disk_max_bytes if disk_max_bytes > 0 else 5 * max_bytes
        # Saves queued by insert_cache(checkpoint=True); flushed by flush_pending_saves().
        self._pending: list[tuple[Any, list[int], Path, list]] = []
        if self._disk_dir:
            self._disk_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface — drop-in replacement for LRUPromptCache
    # ------------------------------------------------------------------

    def fetch_nearest_cache(self, model: Any, tokens: list[int]):
        return self._mem.fetch_nearest_cache(model, tokens)

    def insert_cache(
        self,
        model: Any,
        tokens: list[int],
        prompt_cache: list,
        checkpoint: bool = False,
    ) -> None:
        self._mem.insert_cache(model, tokens, prompt_cache, checkpoint=checkpoint)
        if not (checkpoint and self._disk_dir):
            return

        key = self._cache_key(model, tokens)
        cache_path = self._disk_dir / f"{key}.safetensors"
        if cache_path.exists():
            return  # already on disk

        # Deep-copy while Metal inference may still be in flight.  The copy
        # captures a snapshot of Python handles; the actual GPU buffers are
        # shared with the live cache but will not be mutated.  The real GPU→CPU
        # transfer happens later in flush_pending_saves(), when inference is
        # guaranteed to be idle.
        self._pending.append((model, list(tokens), cache_path, copy.deepcopy(prompt_cache)))

    def flush_pending_saves(self) -> None:
        """Write all queued checkpoint snapshots to disk.

        Must be called from the HTTP handler thread AFTER stream_generate
        returns — i.e., when no Metal inference is active.  All MLX/Metal
        operations (mx.savez GPU→CPU blit) happen here, on the calling thread,
        so there is no concurrent Metal access with ongoing inference.
        """
        while self._pending:
            model, tokens, cache_path, snapshot = self._pending.pop(0)
            self._save_to_disk(model, tokens, cache_path, snapshot)

    def load_from_disk(self, model_namespace: tuple) -> int:
        """Load all saved caches matching this model into memory.

        Call once at startup, before the first request is served.
        Returns the number of entries loaded.
        """
        if not self._disk_dir:
            return 0
        # Newest files first so the LRU ends up prioritising recent entries.
        files = sorted(
            self._disk_dir.glob("*.safetensors"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        loaded = 0
        for cache_path in files:
            try:
                cache, metadata = load_prompt_cache(
                    str(cache_path), return_metadata=True
                )
                saved_model = tuple(json.loads(metadata["model_namespace"]))
                if saved_model != model_namespace:
                    continue  # different model or checkpoint
                tokens = json.loads(metadata["tokens"])
                self._mem.insert_cache(
                    model_namespace, tokens, cache, checkpoint=True
                )
                loaded += 1
                logging.info(
                    "KV cache restored: %d tokens from %s (%.1f MB)",
                    len(tokens),
                    cache_path.name,
                    cache_path.stat().st_size / 1024**2,
                )
            except Exception:
                logging.warning(
                    "Failed to restore KV cache from %s", cache_path, exc_info=True
                )
        if loaded:
            logging.info("KV cache restore complete: %d entries loaded", loaded)
        return loaded

    def log_cache_stats(self) -> None:
        self._mem.log_cache_stats()
        if self._disk_dir:
            files = list(self._disk_dir.glob("*.safetensors"))
            disk_bytes = sum(p.stat().st_size for p in files)
            logging.info(
                "Disk KV cache: %d entries, %.1f MB / %.1f MB budget",
                len(files),
                disk_bytes / 1024**2,
                self._disk_max_bytes / 1024**2,
            )

    def close(self) -> None:
        pass  # no background threads to shut down

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_key(self, model: Any, tokens: list[int]) -> str:
        payload = json.dumps([str(model), tokens], separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    def _save_to_disk(
        self,
        model: Any,
        tokens: list[int],
        cache_path: Path,
        prompt_cache: list,
    ) -> None:
        """Serialise prompt_cache to safetensors (calling thread, Metal idle)."""
        meta_path = cache_path.with_suffix(".json")
        try:
            save_prompt_cache(
                str(cache_path),
                prompt_cache,
                metadata={
                    "model_namespace": json.dumps(list(model)),
                    "tokens": json.dumps(tokens),
                    "n_tokens": str(len(tokens)),
                },
            )
            size = cache_path.stat().st_size
            meta_path.write_text(
                json.dumps(
                    {
                        "model_namespace": list(model),
                        "n_tokens": len(tokens),
                        "file_bytes": size,
                    },
                    indent=2,
                )
            )
            logging.info(
                "KV cache saved: %d tokens → %s (%.1f MB)",
                len(tokens),
                cache_path.name,
                size / 1024**2,
            )
            self._evict_disk()
        except Exception:
            logging.exception("Failed to save KV cache to disk")
            cache_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

    def _evict_disk(self) -> None:
        """Remove oldest safetensors files until total usage is within budget."""
        files = sorted(
            self._disk_dir.glob("*.safetensors"),
            key=lambda p: p.stat().st_mtime,
        )
        total = sum(p.stat().st_size for p in files)
        while total > self._disk_max_bytes and files:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink(missing_ok=True)
            oldest.with_suffix(".json").unlink(missing_ok=True)
            logging.info("Evicted disk KV cache: %s", oldest.name)
