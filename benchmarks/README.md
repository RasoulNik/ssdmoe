# Benchmarks

Benchmark and analysis scripts for the streaming MoE inference engine.

All scripts run directly against the model on disk — no HTTP server required.
They are model-agnostic: `build_streamed_model` auto-detects Qwen3 MoE, Nemotron-H,
and any other supported model type from `config.json`. Model constants (layer count,
expert count, bytes per expert) are read automatically from the `--index` file.

---

## Benchmark scripts

### `bench_generate.py` — Throughput sweep

Sweeps `--top-ks` values and reports generation tok/s, peak memory, and expert read bandwidth.

```bash
poetry run python benchmarks/bench_generate.py \
  --model ~/.cache/huggingface/hub/.../snapshots/<hash> \
  --index .run/qwen35-35b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib \
  --prompt "Explain MoE routing." \
  --top-ks 8,6,4,2
```

Key flags: `--moe-impl streamed|pipelined`, `--fused-gate-up`, `--warmup-tokens`, `--draft-model`

---

### `bench_decode.py` — Decode isolation + profiling

Isolates the decode path with optional expert caching and per-component timing breakdown.
Supports a two-pass approach: realistic tok/s first, then forced-sync compute breakdown.

```bash
# Basic decode
poetry run python benchmarks/bench_decode.py \
  --model <path> --index .run/qwen35-35b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib

# With session-window cache
poetry run python benchmarks/bench_decode.py \
  --model <path> --index .run/qwen35-35b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib \
  --expert-cache-strategy session_window_native --expert-window-tokens 2

# Full timing breakdown
poetry run python benchmarks/bench_decode.py \
  --model <path> --index .run/qwen35-35b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib --profile --two-pass
```

Key flags: `--routed-top-k`, `--expert-cache-strategy`, `--profile`, `--two-pass`

---

### `bench_storage.py` — SSD microbenchmarks

Measures raw expert read throughput without model execution. Only needs `--index`.

| Mode | Path tested | Requires |
|------|-------------|----------|
| `expert` | Python ThreadPool `pread()` | `--index`, `--layer` |
| `component` | Native C batch reader | `--index`, `--native-reader` |
| `both` | Both in sequence | `--index`, `--native-reader` |

```bash
poetry run python benchmarks/bench_storage.py \
  --mode component \
  --index .run/qwen35-35b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib \
  --layers all --experts 0,1,2,3,4,5
```

---

### `bench_expert_hit_rate.py` — Window cache hit rate

Records which experts are selected token-by-token, then simulates a sliding-window
resident cache (H=1, H=2, …) to compute the fraction of SSD reads that would be
eliminated. Outputs a `BenchReport` with per-layer breakdown and projected tok/s gains.

```bash
poetry run python benchmarks/bench_expert_hit_rate.py \
  --model <path> --index .run/qwen35-35b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib \
  --tokens 200 --output .run/hit-rate.json
```

Key flags: `--tokens`, `--windows`, `--ssd-gbps` (default 3.4), `--output`

---

### `bench_routing_correlation.py` — Cross-model routing correlation

Runs the same prompt through a small model and a large model, records expert
selections per layer, and measures Jaccard similarity and hit rate. Answers:
"can the small model's routing predict the large model's routing well enough to
prefetch experts?"

```bash
poetry run python benchmarks/bench_routing_correlation.py \
  --model-small <35B-path> --index-small .run/qwen35-35b-expert-index.json \
  --model-large <122B-path> --index-large .run/qwen35-122b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib \
  --tokens 200 --output .run/routing-corr.json
```

Key flags: `--tokens`, `--routed-top-k`, `--ssd-gbps`, `--output`

---

### `bench_next_layer_predict.py` — Next-layer expert prediction

Tests whether expert selections at layer L can predict layer L+1 selections, using
the "same as previous layer" baseline predictor. Quantifies the upper bound of a
layer-skip prefetch strategy.

```bash
poetry run python benchmarks/bench_next_layer_predict.py \
  --model <path> --index .run/qwen35-35b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib \
  --tokens 200 --output .run/next-layer.json
```

---

## Expert data pipeline

Two-step pipeline for offline expert selection analysis across all layers and topics.

### Step 1 — `collect_expert_data.py`

Runs a single prefill forward pass per topic and records per-token expert selections
for every MoE layer. Topics are loaded from `data/topics.jsonl` (10 topics, ~800-1000
words each).

```bash
poetry run python benchmarks/collect_expert_data.py \
  --model <path> \
  --index .run/qwen35-35b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib \
  --tag 35b \
  --out-dir .run/expert-data
```

Key flags: `--tag` (e.g. `35b`, `122b`), `--topics 0 1 2` (subset), `--topics-file` (custom JSONL),
`--max-tokens`, `--top-k`

Output: `.run/expert-data/<tag>/topic_NN_<name>.npz` + `.run/expert-data/topics.json`

### Step 2 — `train_expert_mapper.py`

Loads collected selections from `35b/` and `122b/` subdirectories, trains a
`CoOccurrenceMapper` and `SlidingWindowMapper` per layer, and reports test-set hit
rates at checkpoints. Reads `n_experts` from npz metadata — no hardcoded constants.

```bash
poetry run python benchmarks/train_expert_mapper.py \
  --data-dir .run/expert-data \
  --output .run/expert-mapper-results.json
```

Key flags: `--layers` (subset to report), `--window-sizes`, `--n-experts` (override)

---

### `bench_expert_mapper.py` — Quick online mapper

> **Deprecated for production use.** Prefer the two-step pipeline above.
> Kept for quick interactive sessions (collect + train in one command).

Collects expert selections on the fly (generation loop, not prefill) and trains a
`CoOccurrenceMapper` in the same process.

---

## Experiments (`experiments/`)

One-off research scripts. Not part of the regular benchmark suite.

| Script | Purpose |
|--------|---------|
| `exp_overlap.py` | Measure I/O vs GPU compute overlap potential for pipelining |
| `exp_mixed_k.py` | Test mixed K values per layer (K=1 early + K=2 late, etc.) |
| `exp_cache_study.py` | Trace decode expert selections, simulate window cache, replay reads |

---

## Multi-model support

All benchmarks work with any model supported by `build_streamed_model`.
Adding a new model requires only building its index — no script changes needed.

| Model | `--index` |
|-------|-----------|
| Qwen3.5-35B-A3B | `.run/qwen35-35b-expert-index.json` |
| Qwen3.5-122B-A10B | `.run/qwen35-122b-expert-index.json` |
| Nemotron-H 30B-A3B | `.run/nemotron30b-expert-index.json` |

### Adding a new model

```bash
# 1. Build the expert index
ssdmoe-build-index \
  --model <path> --output .run/<name>-expert-index.json

# 2. Verify raw SSD throughput
poetry run python benchmarks/bench_storage.py \
  --mode component --index .run/<name>-expert-index.json \
  --native-reader .run/libexpert_reader.dylib --layers all

# 3. Baseline throughput
poetry run python benchmarks/bench_generate.py \
  --model <path> --index .run/<name>-expert-index.json \
  --native-reader .run/libexpert_reader.dylib \
  --prompt "..." --top-ks 8,4,2

# 4. Identify bottlenecks
poetry run python benchmarks/bench_decode.py \
  --model <path> --index .run/<name>-expert-index.json \
  --native-reader .run/libexpert_reader.dylib --profile --two-pass
```

---

## Shared library (`lib/`)

Shared code used by all benchmark scripts. No circular imports.

| Module | Contents |
|--------|---------|
| `lib/loader.py` | `ensure_src_path()`, `parse_bytes()`, `save_json()` |
| `lib/index.py` | `IndexConfig`, `load_index_config()` — reads model constants from `expert_index.json` |
| `lib/hooks.py` | `RecordingSwitch`, `install_recording_hooks()` — non-invasive expert selection capture |
| `lib/decode.py` | `prefill()`, `run_decode()` — shared chunked prefill and decode loop |
| `lib/mapper.py` | `CoOccurrenceMapper`, `SlidingWindowMapper` — expert routing prediction |
| `lib/report.py` | `BenchReport`, `Table`, `Row` — structured terminal output + JSON save |

`lib/index.py` is the key to model-agnosticism: all model constants (`n_experts`,
`n_moe_layers`, `bytes_per_expert_per_layer`, etc.) are derived at runtime from the
index file, so no script contains hardcoded values for a specific model.

### `data/topics.jsonl`

10 topics (~800-1000 words each) used by `collect_expert_data.py`. Pass `--topics-file`
to substitute your own JSONL file (`{"id": N, "name": "...", "text": "..."}`).
