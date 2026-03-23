# MLX Qwen Local Server

This repo runs Qwen 3.5 MLX models locally with `mlx-lm` and exposes an OpenAI-compatible HTTP API for chat UIs and other clients.

## Requirements

- Apple Silicon Mac
- Python 3.10+ available to Poetry
- Poetry installed

## Install

If the environment does not exist yet:

```bash
poetry env use "$(pyenv which python)"
poetry install
```

If you need to add the runtime dependency again:

```bash
poetry add mlx-lm
```

Poetry uses the repo-local environment at `.venv`.

## Start the server

Start the MLX server in the background:

```bash
./scripts/mlx-server.sh start
```

Start the 9B variant explicitly:

```bash
./scripts/mlx-server.sh start 9b
```

Restart the server after changing defaults:

```bash
./scripts/mlx-server.sh restart
```

Restart onto the 9B variant:

```bash
./scripts/mlx-server.sh restart 9b
```

Default settings:

- Default model: `mlx-community/Qwen3.5-4B-MLX-4bit`
- Optional preset: `9b` -> `mlx-community/Qwen3.5-9B-MLX-4bit`
- Host: `127.0.0.1`
- Port: `9000`
- Base URL: `http://127.0.0.1:9000/v1`
- Default max tokens: `16384`
- Thinking: enabled by default

## Check status

```bash
./scripts/mlx-server.sh status
```

## Stop the server

```bash
./scripts/mlx-server.sh stop
```

## Logs

The background server writes logs to:

```bash
.run/mlx-lm.log
```

## API usage

List models:

```bash
curl http://127.0.0.1:9000/v1/models
```

Send a chat request:

```bash
curl http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Qwen3.5-4B-MLX-4bit",
    "messages": [{"role":"user","content":"Write a haiku about MLX."}]
  }'
```

## Open WebUI

Run Open WebUI on the host without Docker:

```bash
./scripts/open-webui.sh start
./scripts/open-webui.sh status
./scripts/open-webui.sh stop
```

Defaults:

- Open WebUI URL: `http://127.0.0.1:3001`
- MLX backend URL: `http://127.0.0.1:9000/v1`
- Open WebUI log: `.run/open-webui.log`

The launcher uses its own virtual environment in `.open-webui-venv` and stores app data in `.open-webui-data`.

## Benchmark

Run the benchmark used in this repo:

```bash
poetry run python -m mlx_lm benchmark \
  --model mlx-community/Qwen3.5-9B-MLX-4bit \
  -p 512 \
  -g 512 \
  -n 3
```

Measured on this machine:

- Prompt throughput: `166.284 tokens/sec`
- Generation throughput: `19.820 tokens/sec`
- Peak memory: `5.734 GB`
