# Model Comparison: Qwen3.5-35B-A3B vs Nemotron-H 30B-A3B

Benchmarked on Apple M4 MacBook Air, 16 GB unified memory, internal SSD (~5.6 GB/s sustained).
Both models run via on-demand expert streaming — no full model load.

---

## Architecture

| | Qwen3.5-35B-A3B | Nemotron-H 30B-A3B |
|--|-----------------|-------------------|
| `model_type` | `qwen3_moe` | `nemotron_h` |
| Total parameters | 35B | 30B |
| Active parameters | 3B | 3B |
| Total layers | 40 | 52 |
| **Mamba (SSM) layers** | 0 | **26** |
| **Attention layers** | 40 | **3** |
| MoE layers | 40 | 23 |
| Attention type | GQA (16 heads / 2 KV) | GQA (32 heads / 2 KV) |
| Experts per MoE layer | 256 | 128 |
| Default top-K | 8 | 6 |
| Expert activation | SwiGLU (gate + up + down) | relu² (fc1 + fc2, no gate) |
| Expert input dim | hidden=2048 | hidden=2688 |
| Expert intermediate dim | 512 | 1856 |
| Shared expert per layer | Yes | Yes |
| Quantization | 4-bit affine (group=64) | 4-bit affine (group=64) |
| Max context | 40,960 tokens | 262,144 tokens |
| Multi-token prediction | No | No |
| KV cache grows with context | Yes (all 40 layers) | **Minimal (3 layers only)** |

**Key structural difference:** Nemotron replaces 37 of 40 attention layers with Mamba SSM. Mamba has no KV cache and O(1) per-token cost regardless of context length — all 26 Mamba layers run identically whether the context is 100 or 100,000 tokens. The KV cache only grows for the 3 attention layers.

---

## Model size and memory

| | Qwen3.5-35B-A3B | Nemotron-H 30B-A3B |
|--|-----------------|-------------------|
| Total on disk (4-bit) | ~19 GiB | ~16.6 GiB |
| Routed expert weights | ~17.8 GiB | ~15.4 GiB |
| Non-expert (resident) | ~1.2 GiB | ~1.17 GiB |
| **Active RAM at runtime** | **~1–2 GB** | **~1.6–1.8 GB** |
| Peak memory (3.6k-token prompt) | ~2.5 GB | ~4.3 GB |

Both models are entirely feasible on 16 GB unified memory. The non-expert weights (attention layers, norms, embeddings) stay resident; routed experts are streamed per token.

---

## Expert sizing: why Nemotron reads more per token

Both models activate ~3B parameters per token, but the *streamed* portion differs significantly:

| | Qwen K=4 | Qwen K=8 | Nemotron K=6 | Nemotron K=3 |
|--|---------|---------|------------|------------|
| Expert size | 1.8 MiB | 1.8 MiB | 5.7 MiB | 5.7 MiB |
| **SSD read/token** | **285 MiB** | 570 MiB | **780 MiB** | **390 MiB** |
| Streamed active params | 0.50B | 1.01B | 1.38B | 0.69B |

Nemotron experts are 3× larger (intermediate_size 1856 vs 512) and each token activates more expert FLOPs despite Nemotron having fewer MoE layers (23 vs 40). The "3B active" claim is accurate for both models but counts different things:

- **Qwen:** 0.50B streamed (SSD) + 2.5B resident (large attention stack)
- **Nemotron:** 1.38B streamed (SSD) + 0.84B resident (Mamba + 3 attention layers)

---

## Throughput benchmarks

### Decode tok/s (short prompt, ~20 tokens)

| K | Qwen | Nemotron |
|---|------|---------|
| 8 / 6 (default) | — | 3.84 tok/s |
| 4 | ~10–12 tok/s (warm) | 4.81 tok/s |
| 3 | — | **5.74–6.79 tok/s** |

*Qwen warm figures reflect OS page-cache hot-caching expert shards during long sessions.*

### Decode tok/s (3,670-token README prompt)

| Model | K | tok/s |
|-------|---|-------|
| Qwen | 4 | ~10 tok/s (warm) |
| Nemotron | 6 | 5.36 tok/s |
| Nemotron | 3 | **5.25 tok/s** |

### Prefill tok/s (3,670-token prompt)

| Model | K | Prefill tok/s | Peak memory |
|-------|---|--------------|-------------|
| Qwen | 4 | ~52 tok/s (server) / ~75 tok/s (bench) | ~2.5 GB |
| **Nemotron** | **3** | **57 tok/s** | 4.3 GB |

Nemotron prefill is faster despite reading more expert data — the 26 Mamba layers have no KV cache to update, making the non-attention layers essentially free during prefill.

---

## Timing breakdown at K=3, decode phase (two-pass profiled)

| Component | Nemotron K=3 | Notes |
|-----------|-------------|-------|
| GPU sync + remap | 64.1 ms (33%) | Constant per layer regardless of K |
| SSD reads | 46.4 ms (24%) | Scales linearly with K |
| qmm_up (GPU) | 29.0 ms (15%) | Compute visible at low K |
| qmm_down (GPU) | 21.5 ms (11%) | |
| bytes→mx.array | 18.7 ms (10%) | Scales with bytes read |
| swiglu/relu² (GPU) | 11.1 ms (6%) | |
| non-MoE (attn+Mamba) | 5.7 ms (3%) | Very low: only 3 attention layers |
| **TOTAL** | **196 ms** | **~6.8 tok/s** |

At K=3, GPU sync is now the #1 bottleneck (33%) — SSD halved but sync remained flat because it scales with MoE layer count (23 layers), not K. K=3 is the sweet spot on this hardware: below this, further K reduction gives diminishing returns as sync dominates.

---

## SSD bandwidth

| Benchmark | Nemotron 30B |
|-----------|-------------|
| Raw component throughput (K=6, 23 layers, bench_storage) | **7.0 GiB/s aggregate** |
| Per-layer best / mean | 1.69 ms / 4.46 ms |
| During decode at K=6 | 10.1 GB/s |
| During decode at K=3 | 7.85 GB/s |

---

## Long-context behaviour

This is Nemotron's primary structural advantage:

| Scenario | Qwen | Nemotron |
|----------|------|---------|
| KV cache grows with context? | Yes — all 40 layers | **No — only 3 layers** |
| KV cache disk cost per token | ~27 KB/token | **~2 KB/token** (3/40 of Qwen) |
| Decode speed at 64K context | Slower (large KV cache) | **Same as short context** |
| Max safe context (16 GB) | ~40K tokens | **>100K tokens** |

At long context, Mamba's O(1) per-token memory and compute means Nemotron maintains constant decode speed while Qwen's attention layers progressively slow down.

---

## When to use which model

| Scenario | Recommended |
|----------|------------|
| Short conversations, max quality | **Qwen K=4** (10–12 tok/s warm) |
| Tool calling, multi-turn chat | **Qwen** (tested, thinking mode) |
| Long context (>20K tokens) | **Nemotron** (KV cache stays small) |
| Speed over quality (quick answers) | **Nemotron K=3** (6.8 tok/s) |
| Throughput-sensitive workloads | **Nemotron K=3** |
| Reasoning / thinking mode | **Qwen** (Qwen3 thinking template) |

---

## Ports and launch scripts

| Model | Script | Default port | Default K |
|-------|--------|-------------|----------|
| Qwen3.5-35B-A3B | `./scripts/streamed-qwen-server.sh` | 9002 | K=4 |
| Nemotron-H 30B-A3B | `./scripts/nemotron-server.sh` | 9003 | K=6 |

Both scripts share a mutex lock (`.run/moe-server.lock`) — starting one while the other is running is blocked to prevent OOM on 16 GB.
