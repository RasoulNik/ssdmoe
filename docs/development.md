# Development Guide

## Prerequisites

- Apple Silicon Mac (M-series), macOS 14+
- 16 GB+ unified memory
- Python 3.10+, [Poetry](https://python-poetry.org/)
- Xcode command-line tools (`xcode-select --install`)
- `mlx-community/Qwen3.5-35B-A3B-4bit` downloaded (~18.5 GB)

## Setup

```bash
git clone <repo>
cd streaming-qwen-server

# Install Python dependencies
poetry install

# Build the native C expert reader library
make -C native
make -C native install   # copies dylibs to .run/

# Build the expert byte-offset index (one-time, ~30 seconds)
ssdmoe-build-index \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/<hash> \
  --output .run/qwen35b-expert-index.json
```

## Starting the Server

```bash
./scripts/streamed-qwen-server.sh start
./scripts/streamed-qwen-server.sh status
./scripts/streamed-qwen-server.sh stop
```

Default endpoint: `http://127.0.0.1:9002/v1`

## Running Benchmarks

```bash
# Decode throughput sweep (K=2,4,6,8)
poetry run python benchmarks/stream_qwen_bench.py \
  --model ~/.cache/.../snapshots/<hash> \
  --index .run/qwen35b-expert-index.json \
  --top-ks 8,6,4,2 --max-tokens 64 \
  --output .run/bench.json

# Decode with session-window expert cache
poetry run python benchmarks/bench_decode_window_cache.py \
  --model ~/.cache/.../snapshots/<hash> \
  --index .run/qwen35b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib \
  --decode-top-k 4 --max-tokens 64

# Component loading breakdown
poetry run python benchmarks/bench_component_loading.py \
  --index .run/qwen35b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib
```

## Verifying After Changes

```bash
poetry run python benchmarks/verify_changes.py \
  --model ~/.cache/.../snapshots/<hash> \
  --index .run/qwen35b-expert-index.json \
  --native-reader .run/libexpert_reader.dylib
```

## Native Library

The native C library (`native/libexpert_reader.dylib`) handles concurrent `pread()`
calls, aligned slab allocation, and batch expert copies.

```bash
make -C native              # builds both dylibs in native/
make -C native install      # copies to .run/
make -C native clean
```

Two builds:
- `libexpert_reader.dylib` — full build with dispatch_apply, slab alloc, copy_experts_multi
- `libexpert_reader_serial.dylib` — serial fallback (read_component_batch only)

## Re-establishing State After Reboot

Run these to confirm the environment:

```bash
date '+%Y-%m-%d %H:%M:%S %Z'
df -h /System/Volumes/Data | sed -n '1,2p'
du -sh ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit
ls .run/qwen35b-expert-index.json .run/libexpert_reader.dylib
```

Key generated artifacts (not committed):
- `.run/qwen35b-expert-index.json` — expert byte-offset index
- `.run/libexpert_reader.dylib` — compiled native reader
- `.run/kv-cache/` — persistent KV cache (safetensors)

## Research Methodology

This project follows a measurement-first approach adapted from `danveloper/flash-moe`:

1. Replace assumptions with local measurements (SSD throughput, expert payload size, tok/s)
2. Keep a running [experiment log](experiment-log.md) — command, artifact, result, conclusion
3. One optimization variable at a time; keep benchmark prompts fixed across comparisons
4. Separate cold-start from steady-state (explicit warmup generation)
5. Prefer official Apple documentation for hardware/OS behavior claims

**Evidence hierarchy:**
- Highest: local measurements, code paths exercised end-to-end
- Medium: microbenchmarks, official Apple docs
- Lower: extrapolations from larger systems or other machines
