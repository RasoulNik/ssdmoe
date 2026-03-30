# Contributing

## Requirements

- Apple Silicon Mac (M-series), macOS 14+
- 16 GB+ unified memory
- Python 3.10+, [Poetry](https://python-poetry.org/)
- Xcode command-line tools
- `mlx-community/Qwen3.5-35B-A3B-4bit` downloaded (~18.5 GB)

## Setup

```bash
poetry install
make -C native && make -C native install
ssdmoe-build-index \
  --model <model_path> --output .run/qwen35b-expert-index.json
```

See [docs/development.md](docs/development.md) for full setup and benchmark instructions.

## Conventions

- Pure Python (`src/streaming_moe/`) — no new external dependencies without discussion
- Hot-path changes (expert loading, decode loop) require a before/after benchmark run
  using `benchmarks/bench_decode_window_cache.py`
- Native C (`native/`) — changes must compile with both `make` targets (dispatch + serial)
- Server changes — verify with a multi-turn opencode session; check `generation stats`
  log lines for cached_tokens regression

## Reporting Issues

Please include:
- macOS version and chip model (`system_profiler SPHardwareDataType | grep -E "Chip|Memory"`)
- Python and mlx-lm versions (`poetry show mlx-lm`)
- Server log excerpt (`.run/streamed-qwen.log`)
