from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

from mlx_lm.generate import stream_generate

from streaming_moe.runtime import build_streamed_model, set_routed_top_k

from .persistent_cache import PersistentPromptCache
from .protocol import build_system_fingerprint


def parse_size(value: str) -> int:
    sizes = {"M": 1_000_000, "G": 1_000_000_000, "MB": 1_000_000, "GB": 1_000_000_000, "": 1}
    split = 0
    for ch in value:
        if not (ch.isdigit() or ch == "."):
            break
        split += 1
    digits = float(value[:split])
    suffix = value[split:].strip().upper()
    return int(digits * sizes[suffix])


class StreamedModelSession:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model_path = Path(args.model).expanduser().resolve()
        self.index_path = Path(args.index).expanduser().resolve()
        self.native_reader_path = (
            Path(args.native_reader).expanduser().resolve()
            if args.native_reader
            else None
        )
        self.decode_top_k = args.routed_top_k
        self.prefill_top_k = args.prefill_top_k or args.routed_top_k
        self.model, self.tokenizer, self.expert_store, _ = build_streamed_model(
            model_path=self.model_path,
            index_path=self.index_path,
            top_k=args.routed_top_k,
            native_reader_path=self.native_reader_path,
            component_workers=args.component_workers,
            use_prefetch=args.enable_prefetch,
            moe_impl=args.moe_impl,
            fused_gate_up=args.fused_gate_up,
            compile_fused_gate_up=args.compile_fused_gate_up,
        )
        self.model_id = args.served_model_id or f"streamed-moe-k{args.routed_top_k}"
        self.system_fingerprint = build_system_fingerprint(
            self.model_id,
            str(self.model_path),
            args.routed_top_k,
        )
        mem_bytes = parse_size(args.prompt_cache_bytes)
        kv_cache_dir_str = getattr(args, "kv_cache_dir", None) or ""
        kv_cache_dir = (
            Path(kv_cache_dir_str).expanduser().resolve()
            if kv_cache_dir_str.strip()
            else None
        )
        disk_bytes_str = getattr(args, "kv_cache_disk_bytes", "0") or "0"
        disk_bytes = parse_size(disk_bytes_str) if disk_bytes_str.strip("0") else 0
        self.prompt_cache = PersistentPromptCache(
            max_size=args.prompt_cache_size,
            max_bytes=mem_bytes,
            disk_dir=kv_cache_dir,
            disk_max_bytes=disk_bytes,  # 0 → 5× mem_bytes inside PersistentPromptCache
        )
        self.cache_namespace = (self.model_id, str(self.model_path))
        self.lock = threading.Lock()

    def close(self) -> None:
        self.prompt_cache.close()
        prefetch_manager = getattr(self.expert_store, "prefetch_manager", None)
        if prefetch_manager is not None:
            prefetch_manager.shutdown()
        self.expert_store.close()

    def set_top_k(self, top_k: int) -> None:
        set_routed_top_k(self.model, top_k)

    def warmup(self) -> None:
        # Restore disk KV caches first so they're available before the first request.
        n = self.prompt_cache.load_from_disk(self.cache_namespace)
        if n:
            logging.info("Loaded %d KV caches from disk (skipping GPU warmup)", n)
            return

        if self.args.warmup_tokens <= 0:
            return
        logging.info(
            "Warmup start: routed_top_k=%s warmup_tokens=%s",
            self.args.routed_top_k,
            self.args.warmup_tokens,
        )
        with self.lock:
            for _ in stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=self.args.warmup_prompt,
                max_tokens=self.args.warmup_tokens,
            ):
                pass
        logging.info("Warmup complete")
