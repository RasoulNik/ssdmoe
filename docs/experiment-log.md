# Streaming MoE Experiment Log

Date started: 2026-03-20

## Goal

Adapt the `flash-moe` design to a Qwen 3.5 MoE model on a `16 GB` M4 Air without loading the full model into RAM and without creating a second full on-disk copy of expert weights.

## Baseline architecture

Reference baseline:

- `danveloper/flash-moe`

Principles kept from the baseline:

- explicit expert routing
- byte-offset expert access
- `pread()`-style reads from original model files
- trust OS page cache first
- treat `K` as a throughput/quality control
- record timing per phase instead of relying on intuition

## Candidate model

Primary candidate:

- `mlx-community/Qwen3.5-35B-A3B-4bit`

Reason for choosing one model only right now:

- disk is limited
- user explicitly asked to avoid downloading multiple large checkpoints in parallel

## Measurements

### Machine

- Apple M4 MacBook Air
- 16 GB unified memory
- internal 256 GB SSD

### Local SSD read tests

Measured on local Hugging Face model shards:

| Test | Result |
|---|---:|
| Sequential `dd` read | `~2.50 GB/s` |
| Large random `pread()` reads (`64 x 7 MiB`) | `~0.93 GB/s` |

### Relevant implication

For Qwen 3.5 35B A3B, the official routing config is `8 activated experts per token` according to the official Hugging Face model page. That makes `K` a first-class optimization knob for this project.

## Completed work

1. Cloned `flash-moe` locally for direct reference.
2. Documented the machine-specific throughput ceiling in `docs/assessment.md`.
3. Added `tools/build_qwen_moe_index.py` to generate a flash-moe-style expert byte index from original safetensors shards.
4. Added `src/streaming_qwen/expert_store.py` to read experts by byte offset with `pread()`.
5. Added `benchmarks/pread_expert_bench.py` to benchmark routed expert reads directly.
6. Added selective tensor loading utilities to load only non-expert text weights from safetensors.
7. Built a working streamed Qwen 3.5 generation path and benchmark harness.
8. Added a native Objective-C expert read benchmark in `native/expert_read_bench.m`.

## Measured streamed generation results

Primary benchmark file:

- `.run/stream-qwen35b-bench.json`

Measured on `Qwen3.5-35B-A3B-4bit`:

| top-k | generation tok/s | peak memory |
|---|---:|---:|
| 8 | 2.40 | 1.54 GB |
| 6 | 2.78 | 2.76 GB |
| 4 | 3.41 | 2.76 GB |
| 3 | 4.21 | 2.76 GB |
| 2 | 4.74 | 2.76 GB |

Conclusion at this stage:

- lowering routed experts per token is the strongest validated throughput lever so far
- pure cold-path measurements badly understate what a long-lived warmed process can do
- benchmark methodology must distinguish cold-start from warm steady-state

## Where time is currently going

Profiled benchmark file:

- `.run/stream-qwen35b-bench-profiled.json`

Key finding from the profiled run:

- wall-clock time is not dominated by raw disk alone
- important costs are:
  - expert load wall time
  - bytes-to-MLX tensor conversion
  - expert `gather_qmm` compute

For the profiled `top-k=8` run:

- expert load wall time: `~4.46 s`
- conversion time: `~2.15 s`
- expert matvec + activation time: `~2.55 s`

This means the next win is more likely to come from reducing conversion / dispatch overhead or reusing expert materialization, not from assuming a faster SSD alone will solve everything.

## Hardware-specific checks

Local macOS tuning facts observed:

- `vfs.generic.lifs.max_ssd_read_size = 8388608` (`8 MiB`)
- this suggests that larger read groupings are a better fit than many tiny reads

### `F_NOCACHE`

Synthetic expert-read microbenchmark:

- default: `~4.08 GiB/s`
- `F_NOCACHE`: `~4.50 GiB/s`

But end-to-end generation with `F_NOCACHE` was worse:

- `.run/stream-qwen35b-bench-nocache.json`
- `top-k=8`: `1.64 tok/s`
- `top-k=4`: `2.27 tok/s`
- `top-k=2`: `3.33 tok/s`

So `F_NOCACHE` is not a net win for the real workload on this machine.

### In-process expert cache

I also tested the idea that this smaller model might benefit from an explicit expert cache even though `flash-moe` did not.

Result:

- larger cache experiments started pushing the machine into heavy compression / memory pressure
- that path is not currently promising on this `16 GB` Air without much tighter cache control

### Native reader

Comparison on the same layer-0, `K=8` expert payload:

- Python benchmark: `benchmarks/pread_expert_bench.py`
- native benchmark: `native/expert_read_bench.m`

Observed after a warm-cache rebooted run:

- Python mean: `~1.38 ms`
- Native Objective-C mean: `~0.58 ms`

Conclusion:

- moving the expert read hot path out of Python is justified by measurement
- a lower-level implementation is the right next optimization direction

### Native reader integrated into end-to-end generation

Benchmark file:

- `.run/stream-qwen35b-bench-native-reader.json`

Measured results:

| top-k | generation tok/s | expert read GiB/s |
|---|---:|---:|
| 8 | 3.18 | 2.99 |
| 4 | 3.82 | 3.45 |
| 2 | 4.17 | 3.11 |

Compared to the earlier pure-Python read path:

- `K=8` improved from roughly `2.40 tok/s` to `3.18 tok/s`

Conclusion:

- the native read path is a real win
- however, even after fixing the read loop, total throughput is still far from `20-30 tok/s`
- the next hot spots are conversion and expert compute, not the raw read syscall path

### Warm steady-state methodology

The earlier anomalous `K=3` and `K=2` numbers turned out to be a methodology problem:

- a multi-`K` sweep warms the page cache and compiled execution paths as it progresses
- later entries in the sweep can look much faster than a fresh single-`K` run

I added explicit warmup support to `benchmarks/stream_qwen_bench.py`:

- `--warmup-tokens`
- `--warmup-prompt`

This keeps cold-start and warm steady-state runs separate.

### Concurrent native component reads

The first native-reader version still fetched all nine routed-expert components serially from Python. I changed `ExpertStore.read_components_batched()` so routed expert components are fetched concurrently with a small worker pool while still using the native `pread()` batch reader underneath.

Key result:

- profiled warm `K=8` improved from about `3.62 tok/s` to about `5.56 tok/s`
- synchronized expert read time dropped from about `4.88 s` to about `3.38 s`

This is the first optimization that materially changed the steady-state frontier.

### Warm steady-state frontier after concurrent component reads

Benchmark file:

- `.run/stream-qwen35b-bench-native-reader-warm8-v2.json`

Measured on the same fixed prompt with `--warmup-tokens 8`:

| top-k | generation tok/s | peak memory | expert read GiB/s |
|---|---:|---:|---:|
| 8 | 7.20 | 1.54 GB | 7.19 |
| 6 | 10.06 | 1.52 GB | 7.48 |
| 4 | 14.11 | 1.49 GB | 9.54 |
| 3 | 15.82 | 1.48 GB | 9.62 |
| 2 | 17.40 | 1.47 GB | 7.80 |

Important caveat:

- these are warmed multi-`K` sweep numbers
- fresh single-`K` warm runs are typically lower
- treat them as optimistic steady-state, not guaranteed first-request throughput

### Single-`K` warm checks

Useful single-`K` warm artifacts:

- `.run/parallel-components-k8k2-warm-repeat.json`
- `.run/k2-warm32.json`
- `.run/k1-warm.json`
- `.run/k1-warm-b.json`

Selected observations:

- single `K=8`, warm: about `6.17 tok/s`
- single `K=2`, warm: about `11.88 tok/s`
- single `K=2`, warm with longer warmup: about `13.11 tok/s`
- single `K=1`, warm: between about `19.65 tok/s` and `24.94 tok/s`

`K=1` is the first configuration that reaches the user’s aspirational throughput range, but the sample output is obviously degraded / broken, so it is not a quality-preserving default.

### Current bottleneck model

Profiled warm artifact after concurrent component reads:

- `.run/profile-warm-k8-v2.json`
- `.run/profile-warm-k2-v2.json`

Key takeaway:

- at `K=8`, synchronized expert read time is still the single biggest bucket
- at `K=2`, expert read time and expert compute are now comparable
- conversion is no longer the dominant cost after the native reader and concurrent component-read changes

For the profiled warm runs:

- `K=8`
  - read: `~3.38 s`
  - convert: `~0.53 s`
  - expert compute (`qmm + swiglu`): `~1.75 s`
- `K=2`
  - read: `~1.24 s`
  - convert: `~0.15 s`
  - expert compute (`qmm + swiglu`): `~1.37 s`

This means the next meaningful gains should come from:

1. native expert materialization or reusable quantized buffers
2. a fatter fused expert execution boundary
3. only then more aggressive storage / compression experiments

### Native reader queue-depth sweep

I parameterized the concurrent native component-read worker count and measured it directly.

Artifacts:

- `.run/worker-sweep-k8-w1.json`
- `.run/worker-sweep-k8-w2.json`
- `.run/worker-sweep-k8-w3.json`
- `.run/worker-sweep-k8-w4.json`
- `.run/worker-sweep-k8-w6.json`
- `.run/worker-sweep-k8-w9.json`
- `.run/worker-sweep-k2-w1.json`
- `.run/worker-sweep-k2-w2.json`
- `.run/worker-sweep-k2-w3.json`
- `.run/worker-sweep-k2-w4.json`
- `.run/worker-sweep-k2-w6.json`

Observed pattern:

- `K=8`
  - `1` worker: `5.03 tok/s`
  - `2` workers: `6.00 tok/s`
  - `3` workers: `6.39 tok/s`
  - `4` workers: `6.39 tok/s`
  - `6` workers: `6.27 tok/s`
  - `9` workers: `6.20 tok/s`
- `K=2`
  - `1` worker: `10.82 tok/s`
  - `2` workers: `11.32 tok/s`
  - `3` workers: `11.71 tok/s`
  - `4` workers: `11.43 tok/s`
  - `6` workers: `10.64 tok/s`

Conclusion:

- the current queue-depth shoulder is around `3-4` workers
- `3` is the best cross-`K` default
- there is no meaningful headroom left in simply raising read concurrency

### Minor conversion tweak

I replaced several `mx.array(np_buffer)` calls with `mx.asarray(np_buffer)` after a small local microbenchmark showed a slight win on the hot U32 materialization path.

Conclusion:

- cheap and reasonable to keep
- incremental only
- not enough to change the overall bottleneck ranking

### Native materialized view attempt

I tried a bounded native-materialization experiment that reused persistent native output buffers and NumPy views for routed expert components instead of allocating fresh output buffers every call.

Artifacts:

- `.run/native-views-k8.json`
- `.run/native-views-k2.json`

Observed result:

- `K=8`: about `4.30 tok/s`
- `K=2`: about `9.59 tok/s`

Both were worse than the existing concurrent native reader baseline.

Conclusion:

- this particular “native materialized views” implementation is a dead end
- the extra pooling / view reuse did not translate into end-to-end gains
- the next native branch should move further toward a fused execution boundary, not just buffer reuse

I reverted this experiment and confirmed the faster baseline still holds:

- `.run/post-revert-k8.json`

### Compression plausibility check

I tested Apple’s built-in `compression_tool` on representative expert-weight slabs.

Artifacts:

- `.run/sample-expert-slab.bin`
- `.run/sample-expert-slab-32.bin`

Measured on a `12 MiB` slab of routed expert weights:

- `lzfse`
  - ratio: `0.658`
  - decode throughput: `~0.66 GiB/s`
- `lz4`
  - ratio: `0.737`
  - decode throughput: `~1.28 GiB/s`

Measured on a `48 MiB` slab:

- `lzfse`
  - ratio: `0.606`
  - decode throughput: `~1.07 GiB/s`
- `lz4`
  - ratio: `0.689`
  - decode throughput: `~2.63 GiB/s`

Conclusion:

- 4-bit expert weights are still compressible enough that chunked compression is not obviously pointless
- however, stock CPU-side decompression throughput is not yet strong enough to be a clear end-to-end win on its own
- compression remains a secondary experiment until native materialization / fused execution is tighter

### Resident scales and biases experiment

I also tested a hybrid layout that keeps all expert scales and biases resident while still streaming the large quantized expert weights.

Memory trade:

- extra resident expert metadata: about `1.875 GiB`
- resulting peak memory: about `3.5 GiB`

Artifacts:

- `.run/resident-small-warm8-full.json`
- `.run/resident-small-profile-k8.json`
- `.run/resident-small-profile-k2.json`

Result:

- mixed
- warm `K=8` improved to about `8.17 tok/s`
- lower-`K` cases did not improve consistently
- synchronized low-`K` profiles suggest extra memory pressure can offset the saved reads

Conclusion:

- keep this mode as an experiment, not the default path

## Open hypotheses

1. A native expert materialization path or reusable quantized buffers should beat the current Python/NumPy-to-MLX conversion materially.
2. A fused Metal or MLX-extension expert boundary should help more than further syscall tuning, especially at lower `K`.
3. Repacking or bundling expert reads closer to the `8 MiB` SSD-friendly region may still help if done without creating a second full model copy.
4. Chunked compression is worth testing only after the compute/materialization path is tighter enough that I/O is clearly the limiter again.
5. Lowering `K` remains the strongest throughput knob, but `K=1` currently breaks output quality.

## Experiments continued (session 2)

### I/O and GPU overlap potential measurement

Measured whether background I/O can complete while GPU is busy.

Artifacts:
- `.run/overlap-experiment.json`

Key findings:
- **1.4-1.7x speedup** achievable by running I/O and GPU in parallel
- I/O time: ~0.8-1.7ms per layer depending on K
- GPU time: ~0.5-1.0ms per layer (simple matmul)
- This confirms overlap is feasible

### Pipelined MoE implementation

Implemented pipelined I/O that overlaps expert loading with shared_expert computation.

New files:
- `src/streaming_qwen/pipelined_moe.py`

Key design:
1. Start expert I/O in background thread
2. Compute `shared_expert(x)` while I/O runs (overlapping work)
3. Wait for I/O and compute expert output

Artifacts:
- `.run/baseline-k2.json`
- `.run/pipelined-k2.json`

Measured results:
- Baseline K=2: 8.08 tok/s
- Pipelined K=2: 9.14 tok/s
- **Improvement: ~13%**
- Expert read throughput increased from 2.92 GB/s to 4.30 GB/s

Conclusion:
- Pipelining provides measurable improvement
- The overlap is working, but the improvement is limited because shared_expert is a small fraction of total work

### ZMLX investigation

Investigated ZMLX (Triton-style kernel toolkit for MLX) for fused MoE kernels.

Key findings:
- ZMLX provides `gather_qmm_swiglu` fused kernel but requires custom MLX build
- Stock MLX 0.31.1 does NOT have `mx.gather_qmm_swiglu`
- ZMLX claims only ~+2% improvement for Qwen3.5-35B on stock MLX
- Not worth the complexity for minimal gain

### Expert compute microbenchmark

Measured isolated expert compute time (gather_qmm x3 + swiglu).

Results:
- K=2: 0.28ms per forward
- K=4: 0.38ms per forward
- K=8: 0.50ms per forward

For 40 layers × 26 tokens:
- K=2: ~291ms theoretical compute
- Actual profiled: ~1.36s

This gap suggests significant overhead beyond just compute:
- Memory allocation/deallocation
- Python loop overhead
- KV cache operations
- Other model components

### mx.compile fusion attempt

Attempted to use `mx.compile` to fuse gate+up+swiglu operations.

Result:
- `GatherQMM cannot infer output shapes` with `shapeless=True`
- Standard compile doesn't work with dynamic shapes from gather operations
- Dead end for this approach

### Fresh baseline (end of session)

Artifact:
- `.run/fresh-baseline-*.json`

Current best results:
| top-k | generation tok/s | peak memory | expert read GB/s |
|---|---:|---:|---:|
| 2 | 9.40 | 1.47 GB | 4.45 |
| 4 | 8.24 | 1.50 GB | 5.37 |
| 8 | 5.83 | 1.54 GB | 4.57 |

## Current-tree SSD vs compute recheck

Artifacts:

- `.run/ssd-vs-compute-current-20260320.json`
- `.run/ssd-vs-compute-profile-20260320.json`

Prompt and settings:

- fixed prompt about storage vs routed-expert compute
- `--warmup-tokens 8`
- native reader enabled
- `component_workers=3`

Fresh non-profiled warm sanity check:

| top-k | generation tok/s | expert read GB/s | peak memory |
|---|---:|---:|---:|
| 8 | 3.22 | 3.00 | 1.57 GB |
| 2 | 7.39 | 3.61 | 1.48 GB |

Fresh profiled run on the same tree:

| top-k | generation tok/s | read seconds | convert seconds | qmm+swiglu seconds |
|---|---:|---:|---:|---:|
| 8 | 2.53 | 6.08 | 1.88 | 3.56 |
| 2 | 4.28 | 1.43 | 0.35 | 2.78 |

Interpretation:

- the current tree is materially slower than the earlier best warm artifacts
- at `K=8`, the routed-expert path is still predominantly read-bound
- at `K=2`, the path is no longer explainable by SSD alone; routed-expert execution is also expensive
- removing storage alone still would not guarantee `20 tok/s` on the current implementation

Useful ceiling math from the indexed expert layout:

- routed expert payload per token:
  - `K=2`: `0.1318 GiB`
  - `K=8`: `0.5273 GiB`
- with the earlier local SSD measurements:
  - `~2.50 GB/s` sequential implies about `18.96 tok/s` at `K=2` and about `4.74 tok/s` at `K=8`
  - `~0.93 GB/s` large-random `pread()` implies about `7.05 tok/s` at `K=2` and about `1.76 tok/s` at `K=8`

Conclusion:

- `K=8` cannot plausibly reach `20 tok/s` on this SSD path without materially reducing bytes per token or depending on a much more resident / compressed layout
- `K=2` is theoretically close on storage alone, but the current routed-expert execution path still leaves a large gap

## Current bottleneck analysis (updated)

At K=2:
- Read: ~1.2s (45% of time)
- Convert: ~0.15s (5% of time)
- Compute: ~1.4s (50% of time)

To reach 20 tok/s from current ~10 tok/s:

## Layer-load parallelism experiment

Artifacts:

- `.run/k8-inner-serial-outer-1.json`
- `.run/k8-inner-serial-outer-3.json`
- `.run/k8-inner-parallel-outer-1.json`
- `.run/k8-inner-parallel-outer-3.json`
- `.run/k2-inner-serial-outer-1.json`
- `.run/k2-inner-serial-outer-3.json`
- `.run/k2-inner-parallel-outer-1.json`
- `.run/k2-inner-parallel-outer-3.json`

Benchmark setup:

- new serial native reader: `native/expert_reader_serial.c`
- layer-load harness: `benchmarks/bench_component_loading.py`
- all `40` text layers
- `5` full sweeps per mode
- compares:
  - inner serial vs inner parallel inside the native reader
  - outer `component_workers=1` vs `3` in `ExpertStore`

Observed aggregate layer-load throughput:

| mode | K=8 GiB/s | K=2 GiB/s |
|---|---:|---:|
| inner serial, outer 1 | 2.68 | 3.79 |
| inner serial, outer 3 | 3.05 | 2.49 |
| inner parallel, outer 1 | 4.13 | 3.65 |
| inner parallel, outer 3 | 4.14 | 2.73 |

Interpretation:

- inner native parallelism is the bigger lever
- at `K=8`, adding outer component parallelism on top of inner native parallelism changes almost nothing
- at `K=2`, extra outer component workers actively hurt
- this matches the structure of the checkpoint:
  - there are `3` large weight streams (`gate`, `up`, `down`)
  - the remaining `6` component reads are only `32 KiB` scales / biases

Practical conclusion:

- keep the native reader parallel inside one component batch
- do not assume more Python-level component workers are better
- once the inner native path is parallel, the next real win should come from fewer larger reads, not more worker fan-out
- Need ~2x improvement
- Pipelining gave ~13%, not enough
- Fused kernels would give ~2-10%, not enough
- Fundamental limit appears to be the combination of I/O + compute

## Gap analysis to 20 tok/s target

Current best K=2: ~10 tok/s (warm)
Target: 20 tok/s

The remaining gap is ~2x. Potential paths:

1. **K=1**: Reaches 20+ tok/s but quality is broken
2. **Aggressive caching**: Would need ~2GB cache, risks memory pressure
3. **Hardware upgrade**: Faster SSD or more memory would help
4. **Model modification**: Fewer layers or smaller experts (quality tradeoff)

## Conclusion for this session

The 20 tok/s target appears to be near the fundamental limit for this hardware + model combination without significant quality degradation. Key findings:

1. **Pipelining works** but limited by the fact that most work can't be parallelized
2. **Fused kernels** provide minimal benefit (~2-10%) on stock MLX
3. **I/O and compute are roughly equal** at K=2, meaning no single optimization can 2x throughput
4. **K=1 reaches target** but breaks quality - the quality/speed tradeoff is steep

The practical recommendation is:
- Use K=2 with pipelined I/O for ~10 tok/s with good quality
- Accept that 20 tok/s may require hardware improvements or quality compromises

## Next experiments

1. Profile the full generation loop to identify remaining overhead sources
2. Measure quality degradation at different K values quantitatively
3. Test if there's a "sweet spot" between K=1 and K=2 using dynamic K selection
4. Investigate if certain layers can use K=1 while others use K=2
5. Consider whether the target should be revised based on hardware constraints

## 2026-03-20: K=4 server usability pass with in-memory KV reuse

Goal:

- move the packaged chat server from `K=2` to `K=4`
- make prompt / KV reuse visible and functional in the OpenAI-compatible server
- verify the behavior with repeated requests, not just direct runtime benchmarks

Implementation:

- default server preset changed to `routed_top_k=4`
- added bounded in-memory prompt cache config:
  - `--prompt-cache-size 8`
  - `--prompt-cache-bytes 1G`
  - `--prefill-step-size 512`
- server now reports `usage.prompt_tokens_details.cached_tokens`
- important MLX detail:
  - Qwen 3.5 currently uses non-trimmable `ArraysCache`
  - storing only the final `prompt + completion` cache entry does not enable
    later prompt-prefix reuse
  - fix: save an explicit `prompt[:-1]` checkpoint during prefill

Files changed:

- `src/streaming_qwen/server/http.py`
- `src/streaming_qwen/server/runtime_adapter.py`
- `src/streaming_qwen/server/protocol.py`
- `scripts/streamed-qwen-server.sh`
- `scripts/streamed-chat.sh`
- `configs/opencode-streamed-simple.json`
- `scripts/opencode-streamed-simple.sh`
- `docs/chat-ui.md`
- `docs/architecture.md`

Direct runtime check after switching to `K=4`:

- `.run/current-direct-k4-short.json`
- `prompt_tps ~= 4.88`
- `generation_tps ~= 3.89`
- `peak_memory ~= 1.45 GB`
- `expert_read_gbps ~= 2.48`

Server-side repeated-request benchmark:

- artifact: `.run/k4-prompt-cache-bench.json`
- clean restart artifact: `.run/k4-prompt-cache-cold-to-warm.json`
- representative results from a warm server:
  - first `turn2`: `prompt_tokens=33`, `cached_tokens=0`, content `"world"`
  - repeated `turn2`: `prompt_tokens=33`, `cached_tokens=32`, content `"world"`
- clean cold-to-warm run:
  - uncached `turn2`: `4.01s`, `cached_tokens=0`
  - repeated `turn2`: `1.29s`, `cached_tokens=32`
  - same content `"world"` in both cases

Observed effect:

- the repeated request now reuses almost the entire prompt from in-memory KV
- server log shows cache growth and live prefix reuse:
  - `KV Caches: 4 seq, 0.15 GB, latest user cache 32 tokens`

Notes:

- the cache path is now functionally correct for repeated prompts and repeated
  chat turns with the same exact prefix
- end-to-end HTTP throughput is still much lower than raw decode throughput
- `opencode` needs another compatibility pass on this branch and should not be
  treated as fully stable yet

## 2026-03-20: OpenCode compatibility pass on the K=4 server

Goal:

- make the packaged streamed server usable from `opencode`
- remove stale config mismatches with the old local `mlx` provider name
- stop SSE hangs caused by long invisible tails after visible output

Implementation:

- repo-level `opencode.json` now defaults to:
  - model: `streamed/streamed-qwen-k4`
  - small model: `streamed/streamed-qwen-k4`
- added a minimal no-tools `streamed-min` agent in `opencode.json`
- fixed `configs/opencode-streamed-simple.json` and
  `scripts/opencode-streamed-simple.sh` to use the `streamed/...` model prefix
- server-side OpenAI compatibility changes:
  - clamp requested completion length to the configured server max
  - keep incremental SSE chunks instead of buffering the whole answer
  - always emit final finish chunk, optional usage chunk, and `[DONE]`
  - stop after a bounded number of invisible generated tokens once visible output
    has started, so clients do not wait forever on chat-template debris

Files changed:

- `opencode.json`
- `configs/opencode-streamed-simple.json`
- `scripts/opencode-streamed-simple.sh`
- `src/streaming_qwen/server/http.py`
- `src/streaming_qwen/server/protocol.py`
- `docs/chat-ui.md`
- `docs/architecture.md`

Verification:

- raw SSE curl request now closes cleanly with:
  - assistant role chunk
  - text chunk(s)
  - final chunk with `finish_reason`
  - usage chunk when requested
  - `[DONE]`
- plain `opencode run` now exits cleanly on the minimal profile:
  - command:
    `opencode run --agent streamed-min --title x --model streamed/streamed-qwen-k4 --format json "Say hello in one short sentence."`
  - observed output:
    - `step_start`
    - `text` with `Hello!`
    - `step_finish`

Notes:

- this is a verified minimal `opencode` path, not full client parity
- tool-enabled agents still fail on this backend because server-side tool calling
  is not implemented yet

## 2026-03-21: Lower-K prefill wins, same-layer prefetch loses

Goal:

- implement the two most credible next-stage optimizations from the routing
  analysis:
  - lower `K` during prompt prefill to improve prefix throughput
  - same-layer next-token speculative prefetch to improve decode throughput
- benchmark them with the same curl-based wall-clock suite:
  - `prefix_100`
  - `prefix_500`
  - `decode_hello160`

Implementation:

- runtime now supports separate `prefill_top_k` and `decode_top_k`
- server prefill path manually runs the prompt prefix at `prefill_top_k`
  and switches back to `decode_top_k` for generation
- streamed switches can optionally use a `PrefetchManager`
- prefetch manager was later tightened to keep only one speculative payload per
  layer after the first unbounded version hit memory exhaustion
- packaged server defaults now keep:
  - `routed_top_k=4`
  - `prefill_top_k=2`
  - `enable_prefetch=false`

Artifacts:

- baseline `K=4 / prefill K=4 / no prefetch`
  - `current-direct-k4-short.json`
  - wall-clock comparison baseline captured during the direct curl run
- `K=4 decode / prefill K=2 / no prefetch`
  - `.run/prefill2-no-prefetch-bench-20260321.json`
- `K=4 decode / prefill K=4 / prefetch`
  - `.run/prefetch-only-bench-20260321.json`
- `K=4 decode / prefill K=2 / prefetch`
  - `.run/prefill2-plus-prefetch-bench-20260321.json`

Results:

- baseline `K=4 / prefill K=4 / no prefetch`
  - `prefix_100`: `12.72 prompt tok/s`
  - `prefix_500`: `25.55 prompt tok/s`
  - `decode_hello160`: `3.47 tok/s`
- `K=4 decode / prefill K=2 / no prefetch`
  - `prefix_100`: `26.59 prompt tok/s`
  - `prefix_500`: `45.15 prompt tok/s`
  - `decode_hello160`: `3.79 tok/s`
- `K=4 decode / prefill K=4 / prefetch`
  - `prefix_100`: `9.93 prompt tok/s`
  - `prefix_500`: `23.24 prompt tok/s`
  - `decode_hello160`: `2.75 tok/s`
- `K=4 decode / prefill K=2 / prefetch`
  - `prefix_100`: `15.99 prompt tok/s`
  - `prefix_500`: `34.58 prompt tok/s`
  - `decode_hello160`: `3.28 tok/s`

Failure mode discovered:

- the first prefetch implementation retained too many speculative payloads and
  crashed on the decode-heavy request with:
  - `malloc: Failed to allocate segment from range group - out of space`
- restricting speculation to one in-flight payload per layer fixed stability
  but did not fix performance

Conclusion:

- lower-`K` prefill is a real win on this machine
- same-layer speculative prefetch is not a win in the current design
- even after fixing the memory blow-up, prefetch regressed both prefix and
  decode wall-clock performance
- the best packaged config remains:
  - `decode K=4`
  - `prefill K=2`
  - `prefetch disabled`

## 2026-03-21: Sliding expert-window study on separate experiment branch

Goal:

- answer a narrower question than end-to-end decode:
  - if we keep the routed experts from the last `H` tokens resident, how much
    SSD traffic can we eliminate?
  - does the realized storage-side gain track the hit ratio, or do smaller miss
    batches degrade SSD efficiency enough to erase the benefit?

Implementation:

- created branch:
  - `exp/speculative-prefetch-limits`
- added lightweight expert-selection tracing in:
  - `src/streaming_qwen/streamed_switch.py`
- added study harness:
  - `benchmarks/speculative_cache_study.py`
- the study:
  - performs a real decode trace with selected experts recorded per layer
  - simulates sliding reuse windows `H=1/2/3`
  - replays only miss sets through the native reader to measure realized
    storage-side improvement

Artifact:

- `.run/speculative-cache-study-20260321.json`

Prompt used:

- `Explain why SSD streaming can bottleneck MoE inference on a laptop, and mention memory pressure too.`

Decode length:

- `24` generated tokens

Key result:

- the user idea is valid, but only in a bounded sliding-window form
- a window of previous-token experts can fit in memory
- the realized benefit depends on routed `K`
- at low `K`, smaller miss batches can become storage-inefficient enough that
  traffic reduction does not translate into faster read time

Top-k `2`:

- baseline replay:
  - `3.296 GiB` read
  - `0.560 s`
  - `5.89 GiB/s`
- `H=1`
  - hit rate: `22.0%`
  - average resident: `0.127 GiB`
  - peak resident: `0.132 GiB`
  - replay time: `0.892 s`
  - result: slower than baseline
- `H=2`
  - hit rate: `28.7%`
  - average resident: `0.220 GiB`
  - peak resident: `0.252 GiB`
  - replay time: `0.686 s`
  - result: still slower than baseline
- `H=3`
  - hit rate: `33.0%`
  - average resident: `0.300 GiB`
  - peak resident: `0.349 GiB`
  - replay time: `0.634 s`
  - result: still slightly slower than baseline

Interpretation for `K=2`:

- low-`K` miss batches become too small
- the SSD/native-reader path loses efficiency
- storage-pattern effects dominate the traffic reduction

Top-k `4`:

- baseline replay:
  - `6.592 GiB` read
  - `1.204 s`
  - `5.48 GiB/s`
- `H=1`
  - hit rate: `25.4%`
  - average resident: `0.253 GiB`
  - peak resident: `0.264 GiB`
  - replay time: `1.291 s`
  - result: slightly slower than baseline
- `H=2`
  - hit rate: `32.7%`
  - average resident: `0.429 GiB`
  - peak resident: `0.506 GiB`
  - replay time: `1.114 s`
  - result: `7.4%` read-time reduction
- `H=3`
  - hit rate: `38.8%`
  - average resident: `0.579 GiB`
  - peak resident: `0.712 GiB`
  - replay time: `1.067 s`
  - result: `11.3%` read-time reduction

Interpretation for `K=4`:

- the plan starts paying off only once the reuse window is larger than one
  token
- `H=2` and `H=3` look plausible on this machine from a memory perspective
- expected end-to-end decode gain will still be smaller than the read-time gain
  because expert compute and conversion remain

Top-k `8`:

- baseline replay:
  - `13.184 GiB` read
  - `2.732 s`
  - `4.83 GiB/s`
- `H=1`
  - hit rate: `31.5%`
  - average resident: `0.506 GiB`
  - peak resident: `0.527 GiB`
  - replay time: `2.050 s`
  - result: `25.0%` read-time reduction
- `H=2`
  - hit rate: `41.3%`
  - average resident: `0.827 GiB`
  - peak resident: `0.995 GiB`
  - replay time: `1.909 s`
  - result: `30.1%` read-time reduction
- `H=3`
  - hit rate: `47.2%`
  - average resident: `1.086 GiB`
  - peak resident: `1.287 GiB`
  - replay time: `1.757 s`
  - result: `35.7%` read-time reduction

Interpretation for `K=8`:

- the same idea becomes clearly beneficial because each layer read stays large
  enough for the storage path to remain efficient
- this is the cleanest evidence that the limiting factor is a combination of:
  - hit ratio
  - and miss-batch SSD efficiency
- not hit ratio alone

Conclusion:

- the right abstraction is a bounded sliding expert working set, not an
  unbounded expert cache
- the proposed reuse plan makes sense
- whether it helps depends strongly on `K`
- for `K=4`, a two- or three-token window is the first point where storage-side
  gains become real
- for `K=2`, the idea is mostly defeated by smaller-miss read inefficiency
- for `K=8`, the idea is clearly valuable even before full end-to-end integration

## 2026-03-21: Integrated exact rolling expert window in the server/runtime

Goal:

- move from the offline study into the real runtime
- integrate an exact rolling expert window into decode
- expose the policy in the server so later branches can test different
  strategies and window sizes
- validate whether the observed storage-side gain produces a comparable
  end-to-end tokens-per-second gain

Implementation:

- added exact per-layer rolling expert reuse to:
  - `src/streaming_qwen/streamed_switch.py`
- added runtime control hooks in:
  - `src/streaming_qwen/runtime.py`
- added server/session control for decode-only activation in:
  - `src/streaming_qwen/server/runtime_adapter.py`
  - `src/streaming_qwen/server/http.py`
- added server flags and wrapper defaults in:
  - `scripts/streamed-qwen-server.sh`

Design:

- cache strategy:
  - `window_exact`
- cache key:
  - exact `(layer, expert)` reuse
- retention:
  - previous `H` decode tokens only
- decode behavior:
  - hits are served from the rolling in-memory expert window
  - misses are read from SSD and then inserted into the window
- prefill behavior:
  - window cache disabled
- request lifecycle:
  - window cache reset at the start and end of each request

Benchmark:

- server config baseline:
  - `routed_top_k=4`
  - `prefill_top_k=2`
  - `expert_cache_strategy=none`
  - `expert_window_tokens=0`
- server config candidate:
  - `routed_top_k=4`
  - `prefill_top_k=2`
  - `expert_cache_strategy=window_exact`
  - `expert_window_tokens=2`
- prompt:
  - `Output the word hello exactly 160 times separated by spaces, and nothing else.`
- request:
  - `max_tokens=256`
  - `temperature=0.0`

Artifacts:

- baseline:
  - `.run/window-cache-baseline-k4.json`
- integrated `H=2` cache:
  - `.run/window-cache-h2-k4.json`

Results:

- baseline:
  - `256` completion tokens in `31.494 s`
  - `8.129 tok/s`
- `window_exact`, `H=2`:
  - `256` completion tokens in `29.270 s`
  - `8.746 tok/s`

Observed gain:

- end-to-end completion throughput improved by about `7.6%`
- elapsed wall time dropped by about `2.224 s`

Interpretation:

- the integrated runtime does benefit from the rolling expert window
- however, the ~`20%` storage-side gain from the offline `K=4, H=2` steady-state
  study does not translate directly into ~`20%` more tok/s
- the remaining gap is expected because end-to-end decode still includes:
  - routed-expert byte-to-MLX conversion
  - expert `gather_qmm` compute
  - server/request overhead

Conclusion:

- this exact rolling `H=2` expert window is a valid improvement and should stay
  as the current `K=4` decode baseline
- it is not enough by itself to deliver a full `20%` throughput gain
- the next gains will need to come from reducing conversion cost or moving more
  of the routed-expert path into a lower-overhead native/Metal execution path

Update after separating cold-start from steady-state:

- the earlier server-level HTTP aggregate was misleading
- a direct decode-only benchmark with per-token timings shows the integrated
  `window_exact, H=2` path is actually slower even after skipping the first
  `2-4` generated tokens

Artifacts:

- direct baseline:
  - `.run/decode-window-baseline-direct.json`
- direct `H=2`:
  - `.run/decode-window-h2-direct.json`

Corrected direct results:

- baseline (`K=4`, `prefill K=2`, no expert window)
  - all tokens: `6.048 tok/s`
  - skip first `2`: `6.102 tok/s`
  - skip first `3`: `6.103 tok/s`
  - skip first `4`: `6.106 tok/s`
  - SSD bytes: `72.76 GB`
  - SSD read time: `15.85 s`
  - effective read throughput: `4.274 GiB/s`
- integrated `window_exact`, `H=2`
  - all tokens: `4.799 tok/s`
  - skip first `2`: `4.850 tok/s`
  - skip first `3`: `4.853 tok/s`
  - skip first `4`: `4.854 tok/s`
  - SSD bytes: `46.71 GB`
  - SSD read time: `16.55 s`
  - effective read throughput: `2.629 GiB/s`

Interpretation:

- the slowdown is not a cold-start artifact
- even in steady state, the current runtime implementation loses
- it does reduce bytes read from SSD, but it makes the remaining read pattern
  too inefficient and still pays the full conversion cost
- the exact rolling-window cache is therefore not yet a winning runtime design
  in its current form

Rerun confirmation on the same branch, one-at-a-time:

- direct baseline rerun:
  - `.run/decode-window-baseline-direct-rerun.json`
  - all tokens: `5.114 tok/s`
  - skip first `2`: `5.151 tok/s`
  - skip first `3`: `5.153 tok/s`
  - SSD bytes: `72.76 GB`
  - SSD read time: `21.06 s`
  - effective read throughput: `3.218 GiB/s`
- direct `H=2` rerun:
  - `.run/decode-window-h2-direct-rerun.json`
  - all tokens: `4.360 tok/s`
  - skip first `2`: `4.399 tok/s`
  - skip first `3`: `4.398 tok/s`
  - SSD bytes: `46.71 GB`
  - SSD read time: `19.77 s`
  - effective read throughput: `2.200 GiB/s`

Conclusion after rerun:

- the A/B result is reproducible even with thermal variation
- the rolling `H=2` cache saves bytes but still loses end-to-end decode throughput
- the most likely cause is the current hit path itself:
  - cached experts are stored as per-expert Python byte blobs
  - the baseline native path converts one contiguous native-read blob per component
  - the windowed path rebuilds component batches from many smaller expert blobs
  - remaining SSD misses also become smaller and less efficient

Memory debugging update:

- `mx.get_peak_memory()` stayed flat at about `1.439 GB` because it only reports MLX
  allocations, not the Python-side rolling expert window
- OS-level memory and new runtime counters show the window cache is real
- short direct decode run, `K=4`, `prefill K=2`, `max_tokens=64`:
  - baseline:
    - `.run/decode-window-baseline-rss64-stats.json`
    - `skip2`: `6.208 tok/s`
    - `window_cache.peak_gib`: `0.0`
  - `window_exact`, `H=2`:
    - `.run/decode-window-h2-rss64-stats.json`
    - `skip2`: `4.051 tok/s`
    - `window_cache.current_gib`: `0.489`
    - `window_cache.peak_gib`: `0.781`
- separate `/usr/bin/time -l` runs showed only a modest RSS delta despite the window
  cache because the baseline already allocates large transient expert blobs during
  decode; the rolling window is not the only significant resident/transient memory
  user of the process

## 2026-03-21: Three targeted optimizations — measured gains and combined impact

### Context

The `window_exact` rolling expert cache (above) was net-negative: it saved 36% of
SSD bytes but lost 20–25% of end-to-end decode throughput.  Root-cause analysis
identified three independent problems that were each responsible for meaningful
overhead.  Each was addressed separately, benchmarked independently, and then
measured as a stack.

All numbers below are **decode-only tok/s** (autoregressive generation, skip first 3
tokens for JIT warmup, SSD stats reset after prefill).  Measurement harness:
`benchmarks/verify_changes.py`.  Model: `Qwen3.5-35B-A3B-4bit`, `K=4`, `H=2`.

---

### Optimization 1 — Native `dispatch_apply` batch reads (replace Python ThreadPool)

**Problem.**  `ExpertStore.read_components_batched` spawned a Python
`ThreadPoolExecutor` with one thread per component (gate, up, down) and called
`read_component_batch` from each thread.  Each call crossed the Python→C ctypes
boundary and paid Python GIL/thread scheduling overhead.  There were
`3 components × 40 layers × N decode tokens` such calls per batch.

**Fix.**  A new C function `read_component_batches` takes a flat list of
`(component, fd, abs_offset, expert_stride, expert_size)` specs and a list of
expert indices, then dispatches all `components × experts` reads in a single
`dispatch_apply` block.  Python makes exactly one ctypes call regardless of how
many components or experts are involved.

**Files changed:**

- `native/expert_io.c` — added `read_component_batches`
- `src/streaming_qwen/native_reader.py` — added ctypes binding
- `src/streaming_qwen/expert_store.py` — `read_components_batched` now calls the
  single-call C path

**Measured result (Test C, microbenchmark, 7.1 MB per call, K=4):**

| Path | Median latency | Throughput |
|------|---------------:|----------:|
| Python ThreadPool (old) | 0.4 ms | 19–20 GB/s |
| Native `dispatch_apply` (new) | 0.1 ms | 50–54 GB/s |
| **Speedup** | **2.57–2.77×** | **2.6–2.7×** |

**End-to-end effect on the no-cache baseline:**
This is already baked into all subsequent baselines.  Comparing the previous
`window_exact` baseline (~6.1 tok/s, K=4) to the new no-cache baseline (13.6–15.7
tok/s) reflects this change alongside the other fixes made in the same commit
window.

---

### Optimization 2 — Session-window native cache: LUT + slab alloc + batch copy

**Problem.**  Even though `window_exact` had the right algorithmic idea, its
Python-side implementation paid three compounding costs per layer-call:

1. **O(K×H) expert lookup** — `_lookup_expert_locked` walked `reversed(session.windows)` for each of K=4 experts; at H=2 that is 8 Python iterations per layer-call × 40 layers × 64 tokens ≈ 20,480 Python dict iterations per batch.
2. **3 separate `malloc` calls per layer** — each layer-call called `alloc_buffer` (ctypes `malloc`) once per component, crossing the ctypes boundary 3 times; 3 × 40 × 64 = 7,680 ctypes alloc+free pairs per decode batch.
3. **Python loop over `ctypes.memmove`** — cache hits were copied component-by-component in a Python loop; at 56% hit rate on K=4 (~2.25 hits/layer) this was ~6.75 memmove calls per layer × 40 layers × 64 tokens ≈ 17,280 ctypes calls per batch.

**Fixes:**

- **O(1) LUT** — `SessionState` now holds `expert_lut: dict[tuple[int,int], tuple[LayerWindow,int]]`; lookup is a single `dict.get` instead of a reverse scan.  Updated on insert and on eviction with an identity guard.
- **Composite slab** — one `posix_memalign(total, 64)` call allocates all components in a single contiguous buffer.  `alloc_slab` / `free_slab` exported from `native/expert_mem.c`.  Python sees a `NativeSlab` object with per-component pointer arithmetic; ctypes boundary crossed once per layer-call instead of once per component.
- **`copy_experts_multi`** — new C function in `native/expert_mem.c` takes arrays of source/destination pointers, expert sizes, and slot indices; dispatches all `components × hits` copies in one `dispatch_apply` block.  Hit path in `load_components_for_layer` makes at most one C call regardless of how many components or experts are cached.

**Files changed:**

- `native/expert_mem.c` — `alloc_slab`, `free_slab`, `copy_experts_multi`
- `src/streaming_qwen/native_reader.py` — `NativeSlab`, ctypes bindings for all three
- `src/streaming_qwen/session_window_cache.py` — `SessionState.expert_lut`, slab-based `LayerWindow`, rewritten `load_components_for_layer`

**C/Python boundary contract per layer-call (after):**

| Call | When | Purpose |
|------|------|---------|
| `alloc_slab(total, 64)` | always | allocate output composite slab |
| `copy_experts_multi(...)` | if any hits | copy all hit experts × all components |
| `read_component_batches_into_slots(...)` | if any misses | read all miss experts × all components |

Down from ~11 ctypes calls per layer-call to at most 3.

**Measured result (K=4, H=2, cold SSD):**

| Path | tok/s | SSD read | Hit rate |
|------|------:|---------:|--------:|
| No cache (baseline) | 8.4 | 18.4 GB | — |
| Session-window H=2 — *before* LUT+slab+copy | 4.6 | 6.1 GB | 56.4% |
| Session-window H=2 — *after* LUT+slab+copy | 7.0 | 7.7 GB | 66.6% |

The LUT+slab+copy fixes brought the cache from **−45% to −18%** versus the no-cache
baseline — a major improvement to the cache path, but it still does not beat the
baseline in raw tok/s.  The cache trades tok/s for SSD traffic: 67% fewer bytes
read, at the cost of ~18% throughput.  The remaining overhead is analysed in the
section below.

---

### Optimization 3 — Fused gate+up projection

**Problem.**  `_compute_expert_output` called `mx.gather_qmm` separately for
`gate_proj` and `up_proj` before combining them with `swiglu`.  These are two
independent linear projections on the same input that could be expressed as a
single operation.

**Fix.**  `src/streaming_qwen/fused_expert.py` provides `compute_expert_output_fused`
which calls `fused_gate_up_swiglu`: one function that performs both gather_qmm
calls and the swiglu activation before handing off to the down projection.
`StreamedSwitchGLU` exposes `fused_gate_up` and `compile_fused_gate_up` flags; the
server passes `--fused-gate-up` through at startup.

**Note on `mx.compile`.** An attempt to compile `fused_gate_up_swiglu` with
`mx.compile(shapeless=True)` crashed at runtime with:
`ValueError: [Primitive::output_shapes] GatherQMM cannot infer output shapes`.
`gather_qmm` output shapes depend on the runtime values of `rhs_indices`, which the
compiler cannot know statically.  The compiled path is permanently disabled;
`use_compiled=False` is the default and the parameter is kept only for API
compatibility.

**Files changed:**

- `src/streaming_qwen/fused_expert.py` — `fused_gate_up_swiglu`, `compute_expert_output_fused`
- `src/streaming_qwen/streamed_switch.py` — `fused_gate_up` / `compile_fused_gate_up` wiring
- `src/streaming_qwen/pipelined_moe.py`, `prefetch_switch.py` — same wiring

**Measured result (Test B, end-to-end decode, K=4, no cache):**

No clean isolated A/B measurement was captured for the fused path alone at the
correct baseline (8.4 tok/s).  The fused and unfused paths were benchmarked under
varying memory pressure conditions and the results cannot be cleanly attributed to
the fused change alone.  The fused gate+up path is structurally correct (two
`gather_qmm` calls replaced by one function call), but its end-to-end impact
at K=4 is within benchmark noise (~1–3%) because the SSD read and attention layers
dominate the token time, not the gate+up compute.  The `mx.compile` path was
permanently disabled — see note above.

---

### Combined gain (cold SSD, K=4, H=2)

Best clean measurement with all fixes applied (cold SSD, no other processes):

| Path | tok/s | SSD read | Hit rate |
|------|------:|---------:|--------:|
| Baseline (no cache) | 8.51 | 22.9 GB | — |
| Session-window H=2 (after all fixes) | 6.97 | 7.7 GB | 66.6% |
| **Delta** | **−18%** | **−67%** | |

The cache consistently saves 67% SSD traffic but is ~18% slower in tok/s.  The
open question — why the cache does not beat the baseline — is the subject of the
next investigation section.

**Root cause of remaining −18% gap:**

Two compounding effects prevent the cache from winning despite 67% SSD savings:

1. **Small-batch SSD inefficiency.**  The baseline reads K=4 experts per layer in one
   `dispatch_apply` batch (3 components × 4 experts = 12 parallel reads).  With 66.6%
   hit rate the miss batch shrinks to ~1.36 experts, yielding ≈4 tasks per call.  The
   sequential-loop threshold in `read_component_batches` kicks in correctly (avoids GCD
   overhead) but each remaining read is individually smaller.  Effective read throughput
   drops from 5.6 GB/s (baseline) to 2.5 GB/s (cached) — the reduction in SSD bytes
   (67%) is almost exactly offset by the reduction in per-byte throughput (55%).
   Net SSD-time benefit: ≈0.8 s saved.

2. **MLX graph overhead from cached hits.**  Cache hits assemble output arrays via
   `mx.take` or `mx.stack` from stored `mx.array` tensors.  These lazy graph nodes add
   to the MLX command graph even though no I/O occurs.  At K=4, H=2, the hit path
   executes `mx.take`/`mx.stack` on ~2.7 layers per token × 40 layers = ~107 extra
   graph nodes per token, adding ~7 µs each for a net +0.75 ms/token overhead at K=4.
   This accounts for roughly the remaining gap after the SSD-time saving.

**Conclusion on session_window_native:**

The sliding expert window cache is net-negative for tok/s at K=4 with H=2 on the M4 Air:
it saves 0.8 s of SSD time per 64-token batch but adds ~7 s of combined cache-lookup and
small-batch read overhead — a net loss.  The design makes more sense at K=8 where both
hit rate and miss-batch sizes are larger.  For the current default K=4 operating point,
`expert_cache_strategy=none` remains the best setting.

**Summary table (corrected):**

| Optimization | Mechanism | Effect |
|---|---|---|
| Native `dispatch_apply` reads | Single C call replaces Python ThreadPool | 2.6–2.8× read latency; baked into all baselines |
| LUT + slab alloc + batch C copy | O(1) lookup, 1 malloc, 1 C copy call per layer | Cache improved from −45% to −18% vs baseline |
| Fused gate+up | Single function for gate+up+swiglu | Within benchmark noise at K=4; SSD dominates |
| Prefill use-after-free fix | Return None early when phase≠decode | Prevents UB; no tok/s effect on decode |

## 2026-03-21: OpenAI tool calling + persistent KV cache

### Tool calling

Goal: enable `opencode` to use the local server with full agentic workflows including
`bash`, `read`, `edit`, and other tools.

Implementation:

- Qwen3.5-35B-A3B uses a proprietary XML tool-call format, not JSON:
  `<tool_call><function=name><parameter=p>value</parameter></function></tool_call>`
- The Jinja chat template also requires `arguments` as a Python dict, not a JSON string —
  passing a JSON string raises `"Can only get item pairs from a mapping"` on multi-turn.
- `visible_stall_tokens=12` was cutting off tool call generation because all tool-call
  tokens are invisible to the stall counter; fixed by raising the stall limit to 2048
  tokens when the request carries a `tools` list.

Files changed:

- `src/streaming_qwen/server/protocol.py` — added `parse_tool_calls()` (XML regex parser),
  `tools` field on `ChatRequest`, `normalize_messages()` now repairs `arguments` from
  JSON string to dict, `prompt_from_messages()` / `prompt_tokens_from_messages()` pass
  `tools=` kwarg to the chat template, `ChatResponseBuilder.chat_completion()` accepts
  `tool_calls=`
- `src/streaming_qwen/server/http.py` — `ServerCapabilities(tools=True)`, `-nothink` model
  suffix detection (sets `enable_thinking=False` without changing the model), stall limit
  bypass for tool requests, post-generation tool-call delta emission in streaming mode
- `opencode.json` — added `streamed-qwen-k4-nothink` model, added `code` agent with
  bash / read / edit / write tools enabled and `steps=20`

Verified end-to-end:

- Non-streaming: `finish_reason: "tool_calls"`, parsed `bash` call with correct arguments
- Streaming: role chunk → tool_call delta → finish chunk
- Multi-turn: model correctly processes tool result and gives final answer
- The `code` agent in opencode routes through this path successfully

### Persistent KV cache

Goal: eliminate cold-start prefill delay; keep prompt KV state across server restarts.

Background: Qwen3.5-35B uses `ArraysCache` (non-trimmable), so mid-conversation prefix
reuse only works if the exact same prefix is re-presented.  In-memory `LRUPromptCache`
already handles the within-session case.  The new disk persistence layer handles the
cross-restart case.

Design:

- Memory budget: `max_bytes` (default 1 GB) via `LRUPromptCache` eviction
- Disk budget: `disk_max_bytes` (default 5 × max_bytes = 5 GB)
- On each conversation checkpoint: deep-copy the KV state, then save to
  `<disk_dir>/<sha256[:20]>.safetensors` in a background thread
- On server startup: `load_from_disk()` restores saved entries; warmup is skipped if
  any entries load successfully
- Lazy `ThreadPoolExecutor`: the background save thread pool is not created until the
  first actual save, avoiding Metal initialization interference during startup

Files changed:

- `src/streaming_qwen/server/persistent_cache.py` (new) — `PersistentPromptCache` wrapping
  `LRUPromptCache` with async safetensors save, `load_from_disk()`, `_evict_disk()`
- `src/streaming_qwen/server/runtime_adapter.py` — replaced `LRUPromptCache` with
  `PersistentPromptCache`, added `load_from_disk()` in `warmup()`, added `close()`
- `src/streaming_qwen/server/http.py` — added `--kv-cache-dir` (default `.run/kv-cache`)
  and `--kv-cache-disk-bytes` server flags

Key engineering note — Metal assertion crash fixed:

Creating `ThreadPoolExecutor` in `__init__` (even without submitting any work) caused
`[_MTLCommandBuffer addCompletedHandler:]:1011` assertions at startup.  Wrapping the
executor creation in `if self._executor is None:` inside `insert_cache()` fixed this.

Verified end-to-end:

- First request: `"KV cache saved: 16 tokens → 04bdd2b3db4ceeaf1d1f.safetensors (31.7 MB)"`
- Server restart: `"KV cache restored: 16 tokens from …safetensors (31.7 MB)"`,
  `"Loaded 1 KV caches from disk (skipping GPU warmup)"`
- Subsequent request: `prompt=17 cached=16 generated=2` — cache hit confirmed

KV cache sizing for reference:

- Qwen3.5-35B-A3B: 341 KB per cached token (40 attention layers, `ArraysCache`)
- 500-token prefix: ~167 MB in memory, ~32 MB on disk after safetensors compression
- Default disk budget (5 GB) can hold ~150 typical conversation prefixes
