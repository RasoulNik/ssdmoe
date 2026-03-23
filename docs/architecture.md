# Streamed Server Architecture

Goal:
- use `mlx_lm.server` as the structural reference
- keep the custom streamed MoE runtime as the generation backend
- make OpenAI-compatible behavior an explicit protocol layer rather than mixed into inference code

Modules:
- `streaming_qwen/server/protocol.py`
  - request normalization
  - capability validation
  - OpenAI-style error envelopes
  - response builders for chat completions and streaming chunks
- `streaming_qwen/server/runtime_adapter.py`
  - owns model/session lifecycle
  - warms the model
  - owns the in-memory LRU prompt/KV cache budget
  - exposes model id and system fingerprint
- `streaming_qwen/server/http.py`
  - HTTP handler and server bootstrap
  - request parsing and dispatch
  - non-stream and SSE chat flows
  - prompt-prefix checkpointing for non-trimmable MLX `ArraysCache`
- `scripts/streamed_qwen_server.py`
  - thin entrypoint only

Why this structure:
- mirrors the separation in `mlx_lm.server` between request handling, generation, and response formatting
- keeps future work local to clear boundaries
- makes it possible to add `tools`, `response_format`, and `/v1/responses` without touching the streamed expert runtime

Current caching behavior:
- the server keeps a bounded in-memory `LRUPromptCache`
- Qwen 3.5 on current MLX uses non-trimmable `ArraysCache`, so storing only
  `prompt + completion` KV state is not enough for prompt-prefix reuse
- the server therefore saves an explicit checkpoint for `prompt[:-1]` during
  prefill, which allows later requests to reuse all but the last prompt token
- the end result is visible in API usage as `prompt_tokens_details.cached_tokens`

Current client-compatibility behavior:

- if a request omits `max_tokens`, the server falls back to its configured default
- SSE streaming emits incremental content chunks, a final finish chunk, an
  optional usage chunk, and then `[DONE]`
- if the model keeps generating invisible template debris after producing visible
  output, the server stops after a bounded visible-stall window so chat clients
  do not hang waiting for a clean stop
- this is enough for the minimal `opencode run` path with an OpenAI-compatible
  provider config

Next feature branches:
1. Tool calling
- add tokenizer capability discovery on startup
- extend request parser for `tools`, `tool_choice`, and `tool` role messages
- add streamed and non-stream `tool_calls` payload emission
- likely borrow the parsing pattern from `mlx_lm.server`

2. Structured outputs
- enable `json_object`
- then add `json_schema` validation/retry policy
- reject unsupported strict schemas explicitly

3. Responses API
- add `/v1/responses` as a translation layer on top of the same runtime adapter
- share the same capability registry and output builders

4. Better client fidelity
- richer error types
- optional logprobs
- more content-part support
- request tracing and optional metrics hooks
