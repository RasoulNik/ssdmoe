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
2. Documented the machine-specific throughput ceiling in `docs/streaming-qwen35b-assessment.md`.
3. Added `scripts/build_qwen_moe_index.py` to generate a flash-moe-style expert byte index from original safetensors shards.
4. Added `streaming_qwen/expert_store.py` to read experts by byte offset with `pread()`.
5. Added `scripts/pread_expert_bench.py` to benchmark routed expert reads directly.
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

- Python benchmark: `scripts/pread_expert_bench.py`
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

I added explicit warmup support to `scripts/stream_qwen_bench.py`:

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
- `streaming_qwen/pipelined_moe.py`

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
- layer-load harness: `scripts/bench_component_loading.py`
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

- `streaming_qwen/server/http.py`
- `streaming_qwen/server/runtime_adapter.py`
- `streaming_qwen/server/protocol.py`
- `scripts/streamed-qwen-server.sh`
- `scripts/streamed-chat.sh`
- `configs/opencode-streamed-simple.json`
- `scripts/opencode-streamed-simple.sh`
- `docs/streamed-chat-ui.md`
- `docs/streamed-server-architecture.md`

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
- `streaming_qwen/server/http.py`
- `streaming_qwen/server/protocol.py`
- `docs/streamed-chat-ui.md`
- `docs/streamed-server-architecture.md`

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
