# SSD MoE — On-Demand Expert Streaming for Large MoE Models on Apple Silicon

**SSD MoE** runs large Mixture-of-Experts (MoE) language models on **any Apple Silicon Mac**
by streaming only the experts needed for each token directly from SSD — no full model load
into RAM, no second copy of weights on disk. Tested on a MacBook Air M4; works with 8 GB, 16 GB, or 24 GB unified memory.

Supported models:

| Model | Parameters | Active/token | Disk | Default K | Port |
|-------|-----------|-------------|------|-----------|------|
| [Qwen3.5-35B-A3B](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit) | 35B | 3B | ~19 GB | 4 | 9002 |
| [Nemotron-H 30B-A3B](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit) | 30B | 3B | ~16.6 GB | 6 | 9003 |

Both models fit within 16 GB unified memory with ~1–2 GB active RAM at runtime; lighter K settings make them viable on 8 GB too.
A shared mutex prevents two models from loading simultaneously (OOM guard).

---

## How SSD Streaming Works

Standard `mlx-lm` loads all model weights into RAM before the first token.
With a 35B model at 4-bit quantization that is ~19 GB — impossible on 16 GB hardware.

SSD MoE exploits MoE sparsity: for every token, only K of the 256 experts per layer
are activated by the router. This server loads **only those K expert shards** per layer
per token, reading them from the original Hugging Face safetensors files with `pread()`:

```
Request
  └─► Tokenise
        └─► KV cache hit? ──yes──► skip prefill, jump to decode
                          ──no───► prefill: pread K experts/layer × N tokens

Decode loop (per token):
  route → pread K expert shards → materialise in MLX → forward pass → emit token

  └─► SSE stream to client
```

Reducing K halves (or quarters) the SSD reads per token, directly trading output quality
for throughput. The server exposes `ROUTED_TOP_K` to tune this at runtime.

---

## Performance

> All numbers measured on **Apple M4 MacBook Air, 16 GB unified memory**, internal SSD (~5.6 GB/s sustained read).
> Throughput scales with SSD bandwidth — expect higher tok/s on Pro/Max chips with faster storage.

### Qwen3.5-35B-A3B (K=4, default)

| Metric | Result |
|--------|--------|
| Decode throughput (fresh process) | **7–8 tok/s** |
| Decode throughput (warm server, page-cache hot) | **10–12 tok/s** |
| Prefill speed — standalone bench | ~75 tok/s |
| Prefill speed — server (step=4096) | ~52 tok/s |
| Multi-turn KV cache hit rate | **97–98%+** |
| Active memory at runtime | ~1 GB |
| Peak memory (3.6k-token prompt) | ~2.5 GB |
| KV cache on disk | ~27 KB / token |

### Nemotron-H 30B-A3B (Mamba-hybrid)

| Metric | K=6 | K=4 | K=3 |
|--------|-----|-----|-----|
| Decode throughput | 3.84 tok/s | 4.81 tok/s | **5.74–6.79 tok/s** |
| SSD reads / token | 780 MiB | 520 MiB | 390 MiB |
| Active memory at runtime | ~1.6–1.8 GB | — | — |
| KV cache on disk | ~2 KB / token | — | — |

Nemotron's Mamba layers (26 of 52 total) have no KV cache — only 3 attention layers grow
with context. Long conversations stay fast where Qwen slows down.

See [docs/model-comparison.md](docs/model-comparison.md) for a full head-to-head analysis
including expert sizing, timing breakdowns, and when to use each model.

---

## Architecture

### Key Components

| Component | Location | Role |
|-----------|----------|------|
| `StreamedSwitchGLU` | `src/streaming_moe/streamed_switch.py` | Replaces mlx-lm MoE layer; SwiGLU (Qwen) or relu² (Nemotron) dispatch |
| `ExpertStore` | `src/streaming_moe/expert_store.py` | Byte-offset index into safetensors shards; `pread()` dispatch |
| `libexpert_reader.dylib` | `native/` | C library: concurrent `pread()`, aligned slab alloc, batch copy |
| `PersistentPromptCache` | `src/streaming_moe/server/persistent_cache.py` | LRU KV cache with safetensors disk checkpoints |
| HTTP server | `src/streaming_moe/server/http.py` | OpenAI-compatible SSE server: tool calling, multi-turn prefix sharing |

### MLX / mlx-lm Components Used

| Symbol | Source | Role |
|--------|--------|------|
| `stream_generate` | `mlx_lm` | Autoregressive decode loop |
| `LRUPromptCache` | `mlx_lm.server` | In-memory KV cache with LRU eviction and prefix trie |
| `make_prompt_cache` | `mlx_lm.models.cache` | Allocates per-request KV state tensors |
| `load_prompt_cache` / `save_prompt_cache` | `mlx_lm.models.cache` | Disk persistence in safetensors format |
| `load_config` / `load_tokenizer` / `_get_classes` | `mlx_lm.utils` | Model config, tokenizer, class resolution at boot |
| `mx.core` / `mlx.nn` | `mlx` | Tensor ops, model layer definitions |

The HTTP server is **not** built on `mlx_lm.server` — it is a custom implementation to
support persistent cross-restart KV caching, tool calling, multi-turn prompt prefix
sharing, and per-request generation statistics.

### Why 97%+ Cache Hits on Multi-Turn (Qwen)

Qwen3's chat template injects `<think>\n\n</think>\n\n` into the **generation prompt**
but omits it from the **history encoding** of completed turns. This creates a token-sequence
mismatch that breaks standard LRU trie walks.

The server stores a second checkpoint after each turn keyed to how that turn will appear
in future history — so turn N+1 finds a 97%+ prefix hit and only prefills the new user
message rather than re-running the full context.

---

## Requirements

- Apple Silicon Mac (M-series), macOS 14+
- 16 GB+ unified memory
- Python 3.10+, [Poetry](https://python-poetry.org/)
- Xcode command-line tools (`xcode-select --install`)

Model checkpoints (download whichever you need):

| Model | HuggingFace repo | Size |
|-------|-----------------|------|
| Qwen3.5-35B-A3B | `mlx-community/Qwen3.5-35B-A3B-4bit` | ~19 GB |
| Nemotron-H 30B-A3B | `mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` | ~16.6 GB |

```bash
pip install huggingface-hub
huggingface-cli download mlx-community/Qwen3.5-35B-A3B-4bit
huggingface-cli download mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit
```

The server reads expert weights **directly from the original safetensors shards** —
no conversion, no second copy needed.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/RasoulNik/ssdmoe.git
cd ssdmoe
poetry install

# 2. Build the native C expert reader
make -C native && make -C native install

# 3. Build the expert byte-offset index (one-time, ~30 s per model)
ssdmoe-build-index \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec \
  --output .run/qwen35b-expert-index.json

ssdmoe-build-index \
  --model ~/.cache/huggingface/hub/models--mlx-community--NVIDIA-Nemotron-3-Nano-30B-A3B-4bit/snapshots/832f602eba5d22436c258c1462bdedc5afddb42b \
  --output .run/nemotron30b-expert-index.json

# 4. Start a server (only one can run at a time — shared OOM guard)
./scripts/streamed-qwen-server.sh start     # Qwen3.5-35B on :9002
./scripts/nemotron-server.sh start          # Nemotron-H 30B on :9003

# 5. Test
curl -s http://127.0.0.1:9002/v1/models | python3 -m json.tool
curl -s http://127.0.0.1:9002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"streamed-moe-k4","messages":[{"role":"user","content":"Hello"}],"stream":false}' \
  | python3 -m json.tool
```

---

## Configuration

Environment variables for both server scripts:

| Variable | Qwen default | Nemotron default | Description |
|----------|-------------|-----------------|-------------|
| `ROUTED_TOP_K` | `4` | `6` | Experts per token — primary throughput/quality knob |
| `PREFILL_TOP_K` | `=ROUTED_TOP_K` | `=ROUTED_TOP_K` | Expert count during prefill |
| `PREFILL_STEP_SIZE` | `4096` | `4096` | Tokens per prefill batch |
| `PROMPT_CACHE_SIZE` | `8` | `8` | Max KV cache entries in memory |
| `PROMPT_CACHE_BYTES` | `1G` | `1G` | Memory budget for in-memory KV cache |
| `COMPONENT_WORKERS` | `3` | `3` | Parallel expert-shard reader threads |
| `HOST` | `127.0.0.1` | `127.0.0.1` | Server bind address |
| `PORT` | `9002` | `9003` | Server listen port |
| `MAX_TOKENS` | `16384` | `16384` | Default max generation tokens |
| `ENABLE_THINKING` | `false` | _(n/a)_ | Qwen3 thinking mode |

Example — start Nemotron at K=3 for maximum throughput:

```bash
ROUTED_TOP_K=3 ./scripts/nemotron-server.sh start
```

Example — start Qwen with persistent KV cache:

```bash
KV_CACHE_DIR=.run/kv-cache ./scripts/streamed-qwen-server.sh start
```

---

## K — The Throughput/Quality Knob

Each token routes through K of the 256 experts per MoE layer. Reducing K lowers SSD
reads per token proportionally:

**Qwen3.5-35B** (40 MoE layers, 3-component SwiGLU experts, 1.8 MiB each):

| K | SSD reads/token | Approx tok/s | Notes |
|---|----------------|--------------|-------|
| 8 | 570 MiB | ~4–5 | Training default, full quality |
| 4 | 285 MiB | **7–12** | Default — good balance |
| 2 | 143 MiB | ~14–16 | Quality degrades noticeably |

**Nemotron-H 30B** (23 MoE layers, 2-component relu² experts, 5.7 MiB each):

| K | SSD reads/token | Approx tok/s | Notes |
|---|----------------|--------------|-------|
| 6 | 780 MiB | ~3.8 | Training default |
| 4 | 520 MiB | ~4.8 | |
| 3 | 390 MiB | **5.7–6.8** | Sweet spot: GPU sync dominates below K=3 |

---

## Using with opencode

[opencode](https://opencode.ai) is an AI coding agent that runs in the terminal.
The repo includes a ready-made config and launcher:

```bash
# Start the Qwen server with persistent KV cache (recommended for long coding sessions)
KV_CACHE_DIR=.run/kv-cache ./scripts/streamed-qwen-server.sh start

# Launch opencode
./scripts/opencode-streamed-simple.sh

# Non-interactive single prompt
./scripts/opencode-streamed-simple.sh run "Explain expert routing in src/streaming_moe/streamed_switch.py"

# Code agent with bash/read/glob/grep tools
OPENCODE_AGENT=code ./scripts/opencode-streamed-simple.sh
```

---

## OpenAI Compatibility

- Streaming (`stream: true`) and non-streaming chat completions
- Tool calling (Qwen3.5 XML format → `tool_calls` in response)
- `cached_tokens` in `usage.prompt_tokens_details`
- System messages, multi-turn history

See [docs/openai-compatibility.md](docs/openai-compatibility.md) for the full checklist.

---

## Repository Layout

```
src/streaming_moe/   Python package — streamed MoE runtime + OpenAI-compatible server
native/               C library — pread, slab allocation, batch expert copy
scripts/              Server launch scripts (.sh) + generation entrypoint
benchmarks/           Decode throughput, storage bandwidth, window cache experiments
  experiments/        Exploratory experiment scripts
  lib/                Shared loader utilities
src/streaming_moe/   Core inference engine + build_index.py (one-time index builder)
docs/                 Architecture, model comparison, experiment log, development guide
configs/              OpenCode agent configuration
```

---

## Documentation

- [Model Comparison](docs/model-comparison.md) — Qwen vs Nemotron: architecture, benchmarks, when to use each
- [Architecture](docs/architecture.md) — server design, caching strategy, module overview
- [Experiment Log](docs/experiment-log.md) — chronological record of measurements and findings
- [Development Guide](docs/development.md) — setup, benchmarks, native build, methodology
- [OpenAI Compatibility](docs/openai-compatibility.md) — supported API features

---

## License

[MIT](LICENSE)
