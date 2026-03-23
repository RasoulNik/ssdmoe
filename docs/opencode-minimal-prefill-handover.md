# OpenCode Minimal Prefill Handover

## Goal

This document is for the next model/backend implementer.

The objective is to run OpenCode against a local OpenAI-compatible backend with the smallest viable prompt/prefill cost on this machine, while still keeping a usable chat path.

## What Caused The "Stuck On Hi" Behavior

The issue was not basic connectivity to the local MLX server.

The main problem was prompt size from OpenCode's default primary agent:

- Default TUI `build` agent on trivial `hi`: about `11,239` prompt tokens
- Lightweight `chat` agent on trivial `hi`: about `1,757` prompt tokens
- `run --agent summary` test path: about `664` prompt tokens

On this Mac, `11k` prompt tokens are enough to make the terminal feel hung even when the backend is healthy.

## Current Working OpenCode Setup

File:

- `opencode.json`

Current defaults:

- Default model: `mlx/mlx-community/Qwen3.5-4B-MLX-4bit`
- Small model: `mlx/mlx-community/Qwen3.5-4B-MLX-4bit`
- Alternate available model: `mlx/mlx-community/Qwen3.5-9B-MLX-4bit`
- Backend URL: `http://127.0.0.1:9000/v1`

Current lightweight primary agent:

- Agent name: `chat`
- Purpose: minimize tool schema + system prompt size
- Allowed tools:
  - `webfetch`
  - `websearch`
- Disabled tools:
  - `bash`
  - `edit`
  - `write`
  - `patch`
  - `task`
  - `todoread`
  - `todowrite`
  - `skill`

Recommended launch command:

```bash
opencode --agent chat
```

If web search is needed:

```bash
OPENCODE_ENABLE_EXA=1 opencode --agent chat
```

For one-shot testing:

```bash
opencode run --agent chat "hello"
```

## Verified Behavior

### MLX backend

The local backend is `mlx_lm server` exposed as an OpenAI-compatible server on port `9000`.

Working command family:

```bash
./scripts/mlx-server.sh start
./scripts/mlx-server.sh start 9b
./scripts/mlx-server.sh restart
./scripts/mlx-server.sh restart 9b
./scripts/mlx-server.sh status
./scripts/mlx-server.sh stop
```

Launcher defaults:

- Default model preset: `4b`
- Optional preset: `9b`
- Thinking: enabled by default

Thinking can be disabled explicitly:

```bash
ENABLE_THINKING=false ./scripts/mlx-server.sh restart
```

### Direct API

Both of these were verified to work against the MLX backend:

- raw `hello` with thinking disabled
- raw `hello` with thinking enabled

With thinking enabled, the backend returned both:

- `message.reasoning`
- `message.content`

### OpenCode chat agent

The lightweight `chat` agent worked against the MLX backend.

Observed result for `hello`:

- final response returned normally
- no "forever stuck" behavior on the minimal agent path

### OpenCode web tools

The `chat` agent is currently configured to allow:

- `webfetch`
- `websearch`

`websearch` requires:

```bash
OPENCODE_ENABLE_EXA=1
```

OpenCode registered both tools successfully when Exa was enabled.

## Prompt/Prefill Observations

This machine is very sensitive to prompt size.

Measured MLX benchmark points:

- `Qwen3.5-9B-MLX-4bit`, `p=512 g=512`: prompt `166.284 tok/s`, generation `19.820 tok/s`
- `Qwen3.5-4B-MLX-4bit`, `p=512 g=512`: prompt `253.936 tok/s`, generation `28.860 tok/s`
- `Qwen3.5-4B-MLX-4bit`, `p=4096 g=512`: prompt `247.269 tok/s`, generation `28.844 tok/s`

Implication:

- the 4B model is the correct default for OpenCode responsiveness on this hardware
- prompt reduction matters more than marginal decode improvements
- reducing tool schema and system prompt size has a large UX effect

## Recommendations For A New Backend

If you are building a replacement backend for OpenCode, optimize for these first:

1. Low prompt/prefill latency on `1k` to `12k` token prompts
2. Reliable streaming responses
3. Normal final content even when reasoning/thinking is enabled
4. Stable OpenAI-compatible chat completions
5. Tool-calling only if actually needed by the selected agent

For minimal-prefill mode, the best strategy is:

1. Keep a lightweight primary agent like `chat`
2. Disable file/shell/task tools unless explicitly needed
3. Default to the smaller model
4. Use a larger model only as an explicit opt-in

## Backend Compatibility Notes

The OpenCode path here assumes an OpenAI-compatible backend.

Minimum practical surface:

- `POST /v1/chat/completions`
- `GET /v1/models`

Important behavior notes:

- OpenCode streams
- OpenCode can become unusable if the backend returns only reasoning with no final content
- large prompts are normal even for simple user messages, depending on agent/tool configuration

## Operational Notes

Useful local files:

- `opencode.json`
- `scripts/mlx-server.sh`
- `.run/mlx-lm.log`
- `.run/opencode-*.log`

If the launcher state becomes stale, check both:

- the PID files in `.run/`
- the actual listener on port `9000`

## Recommended Default User Path

For the best current experience on this repo:

```bash
./scripts/mlx-server.sh start
opencode --agent chat
```

For web search:

```bash
OPENCODE_ENABLE_EXA=1 opencode --agent chat
```
