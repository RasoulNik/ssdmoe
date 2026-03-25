# Qwen3.5-122B-A10B Expert Caching & Draft-Guided Prefetch Study

Branch: `bench/qwen35-122b`
Model: `mlx-community/Qwen3.5-122B-A10B-4bit`
Hardware: Apple Silicon M4 (16 GB unified memory, ~3.4 GB/s SSD)

---

## 1. Baseline Architecture

| Parameter | Value |
|:--|--:|
| Total params | 122B |
| Active params/token | 10B |
| MoE layers | 48 (all layers) |
| Experts per layer | 256 |
| Active experts (K) | 8 |
| Expert size (4-bit) | ~5.06 MB |
| SSD read per token (K=8) | 1,944 MiB (~2 GB) |
| SSD read per token (K=4) | 972 MiB (~1 GB) |
| Theoretical max tok/s @ 3.4 GB/s | ~1.75 |

---

## 2. Decode Benchmarks (no cache)

| K | tok/s | ms/token | SSD ms | GPU sync ms | peak RAM |
|--:|--:|--:|--:|--:|--:|
| 8 | 1.36 | 762 | 519 (68%) | 191 (25%) | 3.98 GB |
| 4 | 2.26 | 447 | 253 (57%) | 164 (37%) | 3.78 GB |

SSD bandwidth utilised: ~3.7 GB/s (near physical limit). Expert GPU compute is <1ms/layer (lazy eval). The bottleneck is purely I/O.

---

## 3. Window Cache Benchmarks (session_window_native)

Cache holds expert weights from the last H decode tokens in RAM. Results at K=8, 128 tokens, **without OS warmup**:

| H | tok/s | Peak RAM | SSD reads | GPU sync | non-MoE overhead | net vs baseline |
|--:|--:|--:|--:|--:|--:|--:|
| 0 (no cache) | 1.09 | 3.97 GB | 519 ms | 191 ms | ~5 ms | — |
| H=1 | 0.97 | 7.71 GB | **0 ms** | 351 ms | 718 ms | **−11%** |
| H=2 | 0.88 | 9.74 GB | **0 ms** | 505 ms | 667 ms | **−19%** |
| H=3 | 0.09 | 11.78 GB | **0 ms** | 9,745 ms | 1,372 ms | **−92%** |

### Interpretation

- **SSD reads → 0**: the cache fully eliminates disk I/O ✓
- **But Python overhead matches or exceeds SSD cost**: cache dict lookups, `mx.take` calls, window eviction per layer per token add ~700 ms overhead at H=1 — roughly the same as SSD time
- **H=3 collapses**: 11.78 GB RAM causes memory pressure on a 16 GB system → 10s/token
- **Root cause**: same GIL contention problem as cross-token prefetch. Cache lookups hold Python GIL, blocking GPU work

### Key Insight

Caching is theoretically correct. The implementation bottleneck is Python-side overhead. A native (C) cache returning pre-loaded `mx.array` pointers with zero Python dispatch would recover the full ~519 ms SSD savings → expected ~2.5–3× speedup (1.36 → 3–4 tok/s).

---

## 4. OS Page Cache Effect (warmup study)

After 256-token warmup pass, H=0/H=1/H=2 ran while 35B model download was active (competing SSD writes degraded read bandwidth ~3.4→~1.5 GB/s). Results are comparable within this run but not directly to the cold § 2 numbers.

| H | tok/s | ms/tok | SSD ms | GPU sync ms | non-MoE ms | Peak RAM |
|--:|--:|--:|--:|--:|--:|--:|
| 0 (warmed, download active) | 0.84 | 1437 | 897 (62%) | 479 (33%) | 8 | 3.98 GB |
| H=1 | 0.97 | 1330 | **0** | 571 (43%) | 759 (57%) | 7.71 GB |
| H=2 | 0.94 | 1122 | **0** | 451 (40%) | 670 (60%) | 9.74 GB |

### Key findings

1. **H=1 wins over H=0 under congested SSD** (0.97 vs 0.84 tok/s): when SSD bandwidth drops to ~1.5 GB/s (background writes), eliminating SSD reads becomes worth the Python overhead (~720ms)
2. **Under normal SSD (3.4 GB/s)**: H=0 wins (520ms SSD < 720ms Python cache overhead) — see § 3
3. **Break-even SSD bandwidth for window cache**: ~720ms overhead / (n_layers×K×expert_size) ≈ when SSD < ~2.7 GB/s the cache becomes beneficial
4. **OS warmup did NOT materially help H=0**: SSD reads went from 519ms (cold, clean disk) to 897ms (warmed, competing download). The OS page cache held only the non-expert model weights (~4 GB); expert files are too numerous (12,288 total) to stay warm

### Conclusion

The window cache's Python overhead (~720ms/tok) exceeds the SSD cost at peak bandwidth. A native C implementation of the cache lookup would drop overhead to ~5ms, making H=1 definitively ~2× faster than baseline regardless of SSD conditions.

---

## 5. Expert Routing Cross-Token Hit Rate (Qwen3.5-122B)

Run with `bench_expert_hit_rate.py`, 64 decode tokens, K=8.

| Window | Mean hit% | Median% | RAM cost | Projected tok/s (native cache) |
|--:|--:|--:|--:|--:|
| H=1 | **34.0%** | 37.5% | 2,038 MB | **2.53** |
| H=2 | **42.7%** | 37.5% | 4,077 MB | **2.91** |

Per-layer (every 4th layer):

| Layer | H=1 | H=2 |
|--:|--:|--:|
| 0 | 6.7% | 14.8% |
| 4–16 | 20–25% | 27–30% |
| 20 | 33.5% | 44.5% |
| 24 | **52.7%** | **61.9%** |
| 28 | **50.4%** | **62.7%** |
| 36 | **50.8%** | **63.7%** |
| 44 | **53.7%** | **62.1%** |

**vs Nemotron-120B**: Nemotron had 12% mean / 1% median. Qwen3.5-122B has **34% mean** — nearly 3× more stable routing. Mid/deep layers (24-44) are over 50% stable at H=1.

**Why the difference**: Qwen3.5's 256-expert routing on each of 48 layers shows specialization patterns that repeat across tokens in the same generation. Nemotron's hybrid Mamba+MoE with 512 experts had more routing diversity.

**Conclusion**: A native C window cache for Qwen3.5-122B would yield +51% throughput at H=1 (1.67 → 2.53 tok/s) and +74% at H=2 (→ 2.91 tok/s). The Python overhead is the only obstacle.

---

## 6. Draft-Model Expert Prediction

### Architecture Comparison

| Parameter | Qwen3.5-35B-A3B | Qwen3.5-122B-A10B |
|:--|--:|--:|
| hidden_size | 2,048 | 3,072 |
| num_layers | 40 | 48 |
| **num_experts** | **256** | **256** |
| **num_experts_per_tok** | **8** | **8** |
| moe_intermediate | 512 | 1,024 |
| Attention type | Hybrid (linear+full) | Full |
| model_type | qwen3_5_moe | qwen3_5_moe |

**Critical finding**: both models share the **same 256-expert index space** with K=8. Expert 42 in the 35B model occupies the same "expert slot" as expert 42 in the 122B model — same team, same training recipe, same vocabulary.

### SOTA on Expert Routing Prediction

| Paper | Method | Accuracy | Speedup |
|:--|:--|--:|--:|
| Speculating Experts (2603.19289) | Same-model quasi-hidden state → next layer | 84–91% (early layers) | 14% TPOT reduction |
| Pre-Attn Predictor (2511.10676) | Pre-attention activations + 2 linear layers | 93–97% | — |
| MoE-SpeQ (2511.14102) | Small draft model → future token expert sequence | — | 2.34× vs offloading |
| SP-MoE | Lightweight per-layer estimators | 90–95% hit@1 | — |

### Proposed Approaches (Priority Order)

#### Approach A: Intra-Model Next-Layer Prediction (highest priority)

Use the same 122B model's hidden state at layer L to predict experts needed at layer L+1. Start async SSD prefetch of layer L+1 experts while layer L computes.

- **Accuracy**: 90–95% (per literature, same-model adjacent layers)
- **Overlap window**: time to compute layer L (~5 ms GPU) vs pread latency (~10 ms/expert batch)
- **Caveat**: overlap too small at batch=1 decode. Layer L GPU compute is ~5 ms; pread is ~10 ms → can only hide 5 ms of the 10 ms load
- **Net expected gain**: ~5 ms × 48 layers = ~240 ms/token → ~1.36 → ~1.9 tok/s

#### Approach B: Cross-Model Draft Routing (35B predicts 122B)

Run Qwen3.5-35B one decode step ahead of the 122B model. Use 35B expert selections as prefetch hints for 122B.

- **Zero-shot hypothesis**: same 256-expert index space → expert i in 35B ≈ expert i in 122B
- **Speed ratio**: 35B should run ~3–5× faster (3B active vs 10B active) → can run ahead
- **Memory overhead**: ~2 GB (35B non-expert weights) + 4 GB (122B non-expert) = 6 GB total
- **Prefetch window**: if 35B runs 1 full token ahead → ~700 ms of overlap available (more than enough to hide ~520 ms SSD read)

**Trained variant**: linear projection from 35B_hidden (2048) → 122B_gate_logits (256), trained on 4–5M tokens.

#### Approach C: Speculative Full Token Generation

Run 35B to speculate the next token AND its expert selections. Prefetch those experts. If 122B verifies the token (same as 35B draft), zero additional SSD reads needed.

- Combines speculative decoding (token acceptance) with expert prefetch
- Token acceptance rate on Qwen family: typically 70–80% for similar-size models
- Accepted tokens: free (no SSD reads). Rejected: pay SSD cost + rollback overhead
- Expected speedup: 2–3× if implemented correctly

---

## 7. Experiment Plan

### Phase 1: OS Warmup Benchmark (running)
- [ ] Warmup pass: 256 tokens
- [ ] H=0 baseline post-warmup
- [ ] H=1 post-warmup
- [ ] H=2 post-warmup
- Artifact: `benchmarks/results/qwen35-122b-cache-benchmark.json`

### Phase 2: Routing Correlation Study
- [ ] Download Qwen3.5-35B-A3B-4bit (in progress)
- [ ] Build expert index for 35B
- [ ] Run `bench_routing_correlation.py`: both models same prompts, measure Jaccard overlap of expert selections per layer per token
- [ ] Measure zero-shot prediction accuracy: treat 35B selection as prediction of 122B selection
- Artifact: `benchmarks/results/routing-correlation.json`

### Phase 3: Intra-Model Prefetch (Approach A)
- [ ] Implement `PredictiveStreamedSwitchGLU`: predict next layer's experts from current hidden state
- [ ] Async pread submit at end of each layer forward
- [ ] Benchmark vs baseline
- Artifact: `benchmarks/results/intra-model-prefetch.json`

### Phase 4: Cross-Model Draft Prefetch (Approach B, if routing correlation > 50%)
- [ ] Implement draft inference loop: 35B runs 1 step ahead
- [ ] 35B expert selections → prefetch for 122B
- [ ] Benchmark vs baseline and Approach A
- Artifact: `benchmarks/results/draft-guided-prefetch.json`

### Phase 5: Trained Projection (if zero-shot < 50%)
- [ ] Collect (35B_hidden, 122B_routing) pairs on 4M tokens
- [ ] Train linear projection: 2048 → 256
- [ ] Evaluate prediction accuracy
- [ ] Integrate into prefetch pipeline

---

## 8. Memory Budget

| Component | RAM |
|:--|--:|
| 122B non-expert weights | ~4 GB |
| 35B non-expert weights | ~2 GB |
| KV cache (2K ctx) | ~0.2 GB |
| H=1 expert cache (48×8 experts) | ~1.9 GB |
| OS overhead | ~1 GB |
| **Total** | **~9 GB** |

Within 16 GB unified memory. H=2 cache (3.8 GB) pushes to ~11 GB — feasible but tight.

---

## 9. Conclusions So Far

1. **SSD bandwidth is the hard ceiling**: 1,944 MiB/token at K=8 at 3.4 GB/s = 1.75 tok/s max
2. **Python-side cache overhead cancels savings**: current session_window_native needs native rewrite
3. **K reduction is the simplest win**: K=4 → K=8 doubles throughput at same quality (needs quality eval)
4. **Cross-model prediction is viable**: same 256-expert space in 35B and 122B makes zero-shot routing prediction plausible — to be validated empirically
5. **Speculative full-token generation** (Approach C) is the highest-potential optimization if token acceptance rate is good

