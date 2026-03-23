# Streamed Qwen K=4 Chat UI

This repo now includes a packaged chat preset for the streamed Qwen 3.5 35B A3B path.

## Packaged backend preset

Launcher:

- `./scripts/streamed-qwen-server.sh`

Default backend settings:

- routed experts per token: `4`
- native reader: `.run/libexpert_reader.dylib`
- component workers: `3`
- prefill step size: `512`
- prompt cache entries: `8`
- prompt cache budget: `1G`
- default max completion tokens: `16384`
- visible-output stall cutoff: `12` generated tokens
- thinking: disabled by default for cleaner chat UI output
- base URL: `http://127.0.0.1:9002/v1`

Commands:

```bash
./scripts/streamed-qwen-server.sh start
./scripts/streamed-qwen-server.sh status
./scripts/streamed-qwen-server.sh stop
./scripts/streamed-qwen-server.sh restart
```

## Web UI

Launcher:

- `./scripts/open-webui-streamed.sh`

Default UI URL:

- `http://127.0.0.1:3002`

Commands:

```bash
./scripts/open-webui-streamed.sh start
./scripts/open-webui-streamed.sh status
./scripts/open-webui-streamed.sh stop
```

Notes:

- the first `start` installs Open WebUI into `.open-webui-venv`
- Open WebUI is configured to use the streamed backend endpoint instead of `mlx-lm`
- if needed, override `STREAMED_BASE_URL` when launching the UI
- prompt-prefix KV cache checkpoints are kept in memory, so repeated requests and
  repeated chat turns with the same prefix can reuse all but the final prompt token

## OpenCode

Repo config:

- `opencode.json`
- default model: `streamed/streamed-qwen-k4`
- minimal no-tools agent: `streamed-min`

Simple helper:

- `./scripts/opencode-streamed-simple.sh`

Verified command:

```bash
cd /Users/rasoul/Documents/projects/MLX
./scripts/streamed-qwen-server.sh start
opencode run --agent streamed-min --title x --model streamed/streamed-qwen-k4 --format json "Say hello in one short sentence."
```

Expected output shape:

- `step_start`
- `text`
- `step_finish`

Notes:

- the streamed backend is OpenAI-compatible enough for the minimal `opencode run` path
- tool-enabled agents are still not supported on this backend
- the server closes SSE cleanly to avoid client hangs

## Verified API checks

Backend verified locally:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Example request:

```bash
curl http://127.0.0.1:9002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "streamed-qwen-k4",
    "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
    "max_tokens": 48,
    "temperature": 0.0
  }'
```
