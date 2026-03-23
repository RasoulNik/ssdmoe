from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

from mlx_lm.generate import stream_generate
from mlx_lm.server import LRUPromptCache

from streaming_qwen.runtime import build_streamed_model, set_routed_top_k

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
        self.model_id = args.served_model_id or f"streamed-qwen-k{args.routed_top_k}"
        self.system_fingerprint = build_system_fingerprint(
            self.model_id,
            str(self.model_path),
            args.routed_top_k,
        )
        self.prompt_cache = LRUPromptCache(
            max_size=args.prompt_cache_size,
            max_bytes=parse_size(args.prompt_cache_bytes),
        )
        self.cache_namespace = (self.model_id, str(self.model_path))
        self.lock = threading.Lock()

    def close(self) -> None:
        prefetch_manager = getattr(self.expert_store, "prefetch_manager", None)
        if prefetch_manager is not None:
            prefetch_manager.shutdown()
        self.expert_store.close()

    def set_top_k(self, top_k: int) -> None:
        set_routed_top_k(self.model, top_k)

    def warmup(self) -> None:
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
