# Server Architecture

## Design goals

- Custom OpenAI-compatible HTTP server (not built on `mlx_lm.server`) to support
  persistent cross-restart KV caching, tool calling, and multi-turn prefix sharing
- Keep the custom streamed MoE runtime as the generation backend
- Make OpenAI-compatible behaviour an explicit protocol layer, separated from inference code

## Module map

| Module | Location | Role |
|--------|----------|------|
| `protocol.py` | `src/streaming_qwen/server/` | Request normalisation, capability validation, OpenAI error envelopes, response builders for chat completions and SSE chunks |
| `runtime_adapter.py` | `src/streaming_qwen/server/` | Model and session lifecycle, model warm-up, in-memory LRU KV cache budget, model ID and system fingerprint |
| `http.py` | `src/streaming_qwen/server/` | HTTP handler and server bootstrap, request dispatch, non-stream and SSE chat flows, prompt-prefix checkpointing, tool-call emission |
| `persistent_cache.py` | `src/streaming_qwen/server/` | Disk KV cache: safetensors checkpoints, LRU eviction, deferred flush, end-of-turn checkpoint for multi-turn prefix sharing |
| `streamed_qwen_server.py` | `scripts/` | Thin entrypoint: arg parsing + `run_server()` |

## KV cache: why two checkpoints per turn

Qwen3's chat template injects `<think>\n\n</think>\n\n` tokens into the generation
prompt when `enable_thinking=False`, but omits them from the history encoding of
completed turns. This creates a token-sequence mismatch: the cache key for turn N
(which includes the injected tokens) does not match turn N+1's prompt prefix.

The server stores two checkpoints after each generation:
1. **Generation cache** — keyed to the full token sequence including injected tokens.
   Used if the exact same generation is re-issued.
2. **End-of-turn checkpoint** — keyed to how this turn will appear in future history
   (without the injected tokens). Used by turn N+1 to get a 97%+ prefix hit.

The end-of-turn checkpoint is stored *before* `insert_cache` is called, because
`insert_cache` can evict the oldest checkpoint (the system+tools base) from memory.

## Prefill loop

The server prefills the prompt in `PREFILL_STEP_SIZE`-token chunks (default 4096)
rather than passing the full prompt to `stream_generate`, because MLX's lazy
evaluation accumulates intermediate Metal buffers across all 94 layers until
`mx.eval()` is called. Chunking + `mx.clear_cache()` between chunks bounds peak
Metal memory to ~5.3 GB for a 4096-token chunk on a 16 GB machine.

## Implemented capabilities

- Streaming SSE and non-streaming chat completions
- Tool calling (OpenAI-compatible `tool_calls` + `tool` role messages)
- Multi-turn conversation with persistent disk KV cache (survives server restart)
- `cached_tokens` in usage response
- Configurable visible-stall guard (stops generation after N invisible tokens)
- `-nothink` model ID suffix to disable thinking mode per request
