# Qwen3.5-35B Streaming MoE Server

Runs **Qwen3.5-35B-A3B** on a **16 GB M4 MacBook Air** via on-demand expert streaming —
no full model load into RAM, no second copy of weights on disk.

**Qwen3.5-35B-A3B is a Mixture-of-Experts (MoE) model.** Each of its 40 transformer layers
contains 256 specialist sub-networks called *experts*. For every token, a learned router
selects K of those 256 experts to run — the other 252 are skipped entirely. With the
training default of K=8, only ~3B of the 35B parameters are active per token. This server
lets you tune K at runtime: lower K = fewer experts loaded from SSD = faster tokens, at some
cost to output quality.

## Performance

| Metric | Result |
|--------|--------|
| Decode throughput (K=4, fresh process) | **7–8 tok/s** |
| Decode throughput (K=4, warm server) | **10–12 tok/s** |
| Prefill speed — cold start (bench) | ~75 tok/s |
| Prefill speed — server (step=4096) | ~52 tok/s |
| Multi-turn KV cache hit rate | **97–98%+** |
| Active memory (server runtime) | ~1 GB |
| Peak memory (during 1.2k-token prefill) | ~2.5 GB |
| Peak memory (during 10k-token prefill) | ~3.9 GB |
| Full model size on disk | ~19 GB |

Platform: Apple M4 MacBook Air, 16 GB unified memory, internal SSD (~5.6 GB/s sustained read).

> Decode throughput scales with OS page cache warmth. A freshly started process reads
> expert shards from SSD on every token (~7–8 tok/s). After a long-running session the
> OS has hot-cached frequently used expert shards in RAM, reaching 10–12 tok/s.
> The server step overhead in prefill (one `mx.clear_cache()` call per 4096-token chunk)
> costs ~30% vs the bench script. `PREFILL_STEP_SIZE=4096` is safe on this machine
> (peak Metal memory 5.3 GB for a single 4096-token chunk vs 16 GB available).

### Long-context prefill and KV cache cost

Measured with a 10,014-token prompt (README + all docs concatenated), K=4:

| | Value |
|---|---|
| Prefill time (bench) | **143 s (2.4 min)** |
| Prefill time (server, step=4096) | **~204 s (3.4 min)** |
| Peak Metal memory during prefill | **3.9 GB** |
| KV cache size on disk | **~265 MB** |

KV cache storage scales at **~27 KB per token**:

| Context | Disk KV cache |
|---------|--------------|
| 1,000 tokens | ~27 MB |
| 4,500 tokens | ~120 MB |
| 10,000 tokens | ~265 MB |
| 32,000 tokens | ~860 MB |

The 3.4-minute cold prefill is a one-time cost per unique conversation prefix.
With 97%+ cache hits, every subsequent turn only prefills the new user message
(typically 50–200 tokens, ~1–4 s) instead of re-running the full context.

## Architecture

Standard `mlx-lm` loads all ~19 GB of model weights into RAM before the first token.
This server loads only the **K routed expert shards** needed for each forward pass, reading
them directly from the Hugging Face safetensors files on SSD with `pread()`.

```
Request
  └─► Tokenise
        └─► KV cache hit? ──yes──► skip prefill, jump to decode
                          ──no───► prefill: pread K experts/layer × N tokens

Decode loop (per token):
  route → pread K expert shards → materialise in MLX → forward pass → next token

  └─► SSE stream to client
```

### Key Components

| Component | Location | Role |
|-----------|----------|------|
| `StreamedSwitchGLU` | `src/streaming_qwen/streamed_switch.py` | Replaces mlx-lm's MoE layer; routes each token to the C reader |
| `ExpertStore` | `src/streaming_qwen/expert_store.py` | Byte-offset index into safetensors shards; `pread()` dispatch |
| `libexpert_reader.dylib` | `native/` | C library: concurrent `pread()`, aligned slab alloc, batch copy |
| `PersistentPromptCache` | `src/streaming_qwen/server/persistent_cache.py` | LRU KV cache with safetensors disk checkpoints |
| HTTP server | `src/streaming_qwen/server/http.py` | OpenAI-compatible SSE server with tool calling + multi-turn prefix sharing |

### MLX / mlx-lm Components Used

| Symbol | Source | How it is used |
|--------|--------|----------------|
| `stream_generate` | `mlx_lm` | Autoregressive decode loop |
| `LRUPromptCache` | `mlx_lm.server` | In-memory KV cache with LRU eviction and prefix trie |
| `make_prompt_cache` | `mlx_lm.models.cache` | Allocates per-request KV state tensors |
| `load_prompt_cache` / `save_prompt_cache` | `mlx_lm.models.cache` | Disk persistence in safetensors format |
| `load_config` / `load_tokenizer` / `_get_classes` | `mlx_lm.utils` | Model config, tokenizer, and class resolution at boot |
| `mx.core` / `mlx.nn` | `mlx` | Tensor ops and model layer definitions |

The HTTP server is **not** built on `mlx_lm.server`; it is a custom implementation to
support persistent cross-restart KV caching, tool calling, multi-turn prompt prefix
sharing, and per-request generation statistics.

### Why 97%+ Cache Hits on Multi-Turn

Qwen3's chat template injects `<think>\n\n</think>\n\n` into the **generation prompt**
(`enable_thinking=False`) but omits it from the **history encoding** of completed turns.
This creates a token-sequence mismatch between turn N's cache key and turn N+1's prompt
prefix — the standard LRU trie walk falls short.

The server stores a second checkpoint after each turn keyed to how that turn will appear
in future history (the "end-of-turn checkpoint"), so turn N+1 finds a 97%+ prefix hit
and only prefills the new user message and tool results.

## Requirements

- Apple Silicon Mac (M-series), macOS 14+
- 16 GB+ unified memory (runs on 16 GB; more headroom is better)
- Python 3.10+, [Poetry](https://python-poetry.org/)
- Xcode command-line tools (`xcode-select --install`)
- `mlx-community/Qwen3.5-35B-A3B-4bit` model checkpoint (~18.5 GB on disk)

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd streaming-qwen-server
poetry install

# 2. Build the native C expert reader
make -C native && make -C native install

# 3. Build the expert byte-offset index (one-time, ~30 s)
poetry run python tools/build_qwen_moe_index.py \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/<hash> \
  --output .run/qwen35b-expert-index.json

# 4. Start the server
./scripts/streamed-qwen-server.sh start

# 5. Test
curl -s http://127.0.0.1:9002/v1/models | python3 -m json.tool
curl -s http://127.0.0.1:9002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"streamed-qwen-k4","messages":[{"role":"user","content":"Hello"}],"stream":false}' \
  | python3 -m json.tool
```

## Configuration

All options are passed via environment variables to `scripts/streamed-qwen-server.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTED_TOP_K` | `4` | Experts loaded per token — the primary throughput/quality knob |
| `PREFILL_TOP_K` | `=ROUTED_TOP_K` | Expert count during prefill (can differ from decode) |
| `PREFILL_STEP_SIZE` | `1024` | Tokens per prefill batch |
| `PROMPT_CACHE_SIZE` | `8` | Max KV cache entries in memory |
| `PROMPT_CACHE_BYTES` | `1G` | Memory budget for in-memory KV cache |
| `KV_CACHE_DIR` | _(none)_ | Directory for persistent disk KV cache |
| `COMPONENT_WORKERS` | `3` | Parallel expert-shard reader threads |
| `HOST` | `127.0.0.1` | Server bind address |
| `PORT` | `9002` | Server port |
| `MAX_TOKENS` | `16384` | Default max generation tokens |

Example — enable persistent KV cache:

```bash
KV_CACHE_DIR=.run/kv-cache ./scripts/streamed-qwen-server.sh start
```

## K — Number of Active Experts

Each token's forward pass routes through K of the 256 experts in every MoE layer.
The model was trained with K=8, which activates ~3B parameters per token and gives
full output quality. This server streams only the K chosen expert shards from SSD on
each token, so reducing K directly reduces both I/O and compute:

```
Per token: 40 layers × K experts × 3 components (gate, up, down) read from SSD
K=8 → 960 expert-component reads / token
K=4 → 480 expert-component reads / token  (2× faster I/O)
K=2 → 240 expert-component reads / token  (4× faster I/O, quality degrades)
```

| K | Active params/token | Fresh-process tok/s | Notes |
|---|--------------------|--------------------|-------|
| 8 | ~3.0 B (trained default) | ~4–5 | Full quality |
| 4 | ~1.5 B | **7–8** | Default here — good balance |
| 2 | ~0.75 B | ~14–16 | Noticeably degraded |

Set via `ROUTED_TOP_K` environment variable before starting the server.

## OpenAI Compatibility

The server implements a subset of the OpenAI chat completions API:
- Streaming (`stream: true`) and non-streaming
- Tool calling (Qwen3.5 XML format, returned as `tool_calls`)
- `cached_tokens` in `usage.prompt_tokens_details`
- System messages, multi-turn history

See [docs/openai-compatibility.md](docs/openai-compatibility.md) for the full checklist.

## Repository Layout

```
src/streaming_qwen/   Python package — streamed MoE runtime + OpenAI server
native/               C library — pread, slab allocation, batch expert copy
scripts/              Server launch scripts (shell) + generation entrypoint
benchmarks/           Decode throughput, component loading, expert cache experiments
tools/                build_qwen_moe_index.py — one-time index generation
docs/                 Architecture, experiment log, development guide
configs/              OpenCode agent configuration
```

## Documentation

- [Architecture](docs/architecture.md) — server design, caching strategy, module overview
- [Experiment Log](docs/experiment-log.md) — chronological record of measurements and findings
- [Development Guide](docs/development.md) — setup, benchmarks, native build, methodology
- [OpenAI Compatibility](docs/openai-compatibility.md) — supported API features

## License

[MIT](LICENSE)
