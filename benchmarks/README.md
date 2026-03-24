# Benchmarks

Benchmark and profiling scripts for the streaming MoE inference server.

All scripts run directly against the model on disk — no HTTP server required.
They are model-agnostic: `build_streamed_model` auto-detects Qwen3 MoE, Nemotron-H,
and any other supported model type from `config.json`.

---

## Scripts

### `bench_generate.py` — Throughput sweep
Sweeps `--top-ks` values and reports generation tok/s, peak memory, and expert read bandwidth.
Use this to characterize model throughput vs quality trade-off (K=4 is the default operating point).

```bash
poetry run python benchmarks/bench_generate.py \
  --model ~/.cache/huggingface/hub/.../snapshots/<hash> \
  --index .run/qwen35b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib \
  --prompt "Explain MoE routing." \
  --top-ks 8,6,4,2
```

Key flags: `--moe-impl streamed|pipelined`, `--fused-gate-up`, `--warmup-tokens`, `--draft-model`

---

### `bench_decode.py` — Decode isolation + profiling
Isolates the decode path (no server overhead), with optional expert caching and per-component
timing breakdown. Supports a two-pass approach: realistic tok/s + forced-sync compute breakdown.

```bash
# Basic decode
poetry run python benchmarks/bench_decode.py \
  --model <path> --index .run/qwen35b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib

# With session-window cache
poetry run python benchmarks/bench_decode.py \
  --model <path> --index .run/qwen35b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib \
  --expert-cache-strategy session_window_native --expert-window-tokens 2

# Full timing breakdown (two-pass: realistic + GPU-synced profiling)
poetry run python benchmarks/bench_decode.py \
  --model <path> --index .run/qwen35b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib --profile --two-pass
```

Key flags: `--routed-top-k`, `--expert-cache-strategy`, `--profile`, `--two-pass`

---

### `bench_storage.py` — SSD microbenchmarks
Measures raw expert read throughput without model execution.
Model-agnostic — only needs `--index`. Two modes:

| Mode | Path tested | Requires |
|------|-------------|----------|
| `expert` | Python ThreadPool `pread()` | `--index`, `--layer` |
| `component` | Native C batch reader | `--index`, `--native-reader` |
| `both` | Both in sequence | `--index`, `--native-reader` |

```bash
# Component mode across all layers (typical)
poetry run python benchmarks/bench_storage.py \
  --mode component \
  --index .run/nemotron30b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib \
  --layers all --experts 0,1,2,3,4,5

# Expert (pread) mode, single layer
poetry run python benchmarks/bench_storage.py \
  --mode expert \
  --index .run/qwen35b-expert-index.json \
  --layer 5
```

---

## Experiments (`experiments/`)

One-off research scripts. Not part of the regular benchmark suite but kept for reproducibility.

| File | Purpose |
|------|---------|
| `exp_mixed_k.py` | Test mixed K values across layers (quality vs speed per-layer) |
| `exp_cache_study.py` | Trace expert selection and replay to study window cache hit rates |
| `exp_overlap.py` | Measure I/O vs GPU compute overlap potential for pipelining |

---

## Multi-model support

All benchmarks work with any model supported by `build_streamed_model`:

| Model | `--index` | Default `--top-ks` |
|-------|-----------|-------------------|
| Qwen3.5-35B-A3B | `.run/qwen35b-expert-index.json` | `8,6,4,2` |
| Nemotron-H 30B-A3B | `.run/nemotron30b-expert-index.json` | `6,4,2` |

Model type (activation, weight paths) is auto-detected from `config.json`.

---

## New model checklist

When adding a new model:
1. Build its expert index: `poetry run python tools/build_moe_index.py --model <path> --output .run/<name>-expert-index.json`
2. Run `bench_storage.py --mode component` to verify the index and measure raw SSD throughput
3. Run `bench_generate.py` to get tok/s baseline
4. Run `bench_decode.py --two-pass` to identify bottlenecks (SSD read vs compute vs other)

---

## Shared library (`lib/`)

| Module | Contents |
|--------|---------|
| `lib/loader.py` | `ensure_src_path()`, `parse_bytes()`, `save_json()` |

All benchmark scripts call `sys.path.insert(0, .../src)` at the top, so they run correctly
whether invoked via `poetry run`, from the repo root, or directly.
