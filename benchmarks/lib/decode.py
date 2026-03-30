"""Shared prefill and decode loop utilities.

Usage:
    from lib.decode import prefill, run_decode
    from lib.loader import ensure_src_path; ensure_src_path()

    cache = make_prompt_cache(model)
    last = prefill(model, tokenizer.encode(prompt), cache)

    # Or run the full loop (attaches any RecordingSwitch hooks before calling):
    tokens = run_decode(model, tokenizer, prompt, n_tokens=64, top_k=8)
"""
from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from lib.loader import ensure_src_path

ensure_src_path()

from streaming_moe.runtime import set_routed_top_k


def prefill(
    model,
    prompt_tokens: list[int],
    prompt_cache,
    step_size: int = 1024,
) -> mx.array:
    """Chunked prefill; returns the last token as a (1,) array.

    Processes prompt[:-1] in chunks of step_size (to bound peak memory),
    calling mx.clear_cache() between chunks.  Returns prompt[-1:] ready to
    pass as the first token to a generate_step loop.
    """
    prompt = mx.array(prompt_tokens, dtype=mx.uint32)
    if prompt.size <= 1:
        return prompt
    remaining = prompt[:-1]
    while remaining.size > 0:
        n = min(step_size, remaining.size)
        model(remaining[:n][None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])
        remaining = remaining[n:]
        mx.clear_cache()
    return prompt[-1:]


def run_decode(
    model,
    tokenizer,
    prompt: str,
    n_tokens: int,
    top_k: int,
    step_size: int = 1024,
) -> list[int]:
    """Prefill prompt then decode up to n_tokens, returning generated token ids.

    Sets routed top_k on the model before running.  If RecordingSwitch hooks
    have been installed on the model, they will fire during the decode loop.
    """
    from mlx_lm.generate import generate_step

    set_routed_top_k(model, top_k)
    tokens = tokenizer.encode(prompt)
    cache = make_prompt_cache(model)
    last = prefill(model, tokens, cache, step_size)
    eos_ids = set(getattr(tokenizer, "eos_token_ids", []) or [])
    generated: list[int] = []
    for token, _ in generate_step(last, model, max_tokens=n_tokens, prompt_cache=cache):
        mx.eval(token)
        tid = int(token)
        generated.append(tid)
        if tid in eos_ids:
            break
    return generated
