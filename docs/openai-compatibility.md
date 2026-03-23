# OpenAI-Compatible Server Checklist

Purpose:

- define the minimum compatibility surface for local clients that expect OpenAI-style APIs
- separate "basic chat works" from "real compatibility"

## Baseline endpoints

Minimum practical endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Recommended next endpoint:

- `POST /v1/responses`

Reason:

- many newer clients and SDK paths are moving toward Responses
- some clients still rely on Chat Completions only

## Response object fidelity

For `chat.completion` responses:

- `id`
- `object`
- `created`
- `model`
- `choices[].index`
- `choices[].message.role`
- `choices[].message.content`
- `choices[].finish_reason`
- `usage.prompt_tokens`
- `usage.completion_tokens`
- `usage.total_tokens`

Better compatibility:

- `choices[].message.refusal`
- `choices[].message.annotations`
- `choices[].logprobs`
- `usage.prompt_tokens_details`
- `usage.completion_tokens_details`
- `service_tier`
- `system_fingerprint`

## Streaming fidelity

For `stream: true` on Chat Completions:

- SSE framing with `data: ...`
- final `data: [DONE]`
- chunk object `object: "chat.completion.chunk"`
- stable `id` across chunks
- `choices[].delta.role` on first assistant chunk
- `choices[].delta.content` token deltas
- `choices[].finish_reason` on final chunk

Better compatibility:

- `stream_options.include_usage`
- final usage chunk with empty `choices`
- `usage: null` on non-final chunks when usage streaming is requested

## Message input fidelity

Input messages should support:

- `system`
- `developer`
- `user`
- `assistant`
- `tool`

Content should support both:

- plain string content
- typed content-part arrays

At minimum for broad chat compatibility:

- text parts

For wider parity:

- image parts
- audio parts where relevant

## Tool calling

Proper compatibility requires:

- request-side `tools`
- request-side `tool_choice`
- response-side `choices[].message.tool_calls`
- streamed `delta.tool_calls`
- support for `tool` role follow-up messages with `tool_call_id`

If supporting strict function calling:

- honor `strict: true`
- validate schemas enough to reject unsupported strict schemas clearly

## Structured outputs

Two separate features matter:

1. JSON mode
- request `response_format: { "type": "json_object" }`
- response content must be valid JSON

2. Structured Outputs
- request `response_format: { "type": "json_schema", "json_schema": ... }`
- enforce schema-shaped output, especially when `strict: true`

This is not the same as "prompt the model to return JSON".
The server must understand the request contract and either:

- implement it correctly
- or reject it clearly

## Error behavior

A proper server should return OpenAI-style JSON errors, not ad hoc strings.

Baseline:

- correct HTTP status
- body shaped like `{ "error": { ... } }`

Useful fields:

- `message`
- `type`
- `param`
- `code`

## Headers and observability

Helpful production headers:

- `x-request-id`
- `openai-processing-ms`
- `openai-version`

Useful if you emulate rate limits:

- `x-ratelimit-limit-*`
- `x-ratelimit-remaining-*`
- `x-ratelimit-reset-*`

## Behavior details that break clients if missing

- exact SSE formatting
- exact finalization behavior for streaming
- correct `finish_reason`
- proper `tool_calls` shape
- `developer` role support
- structured-output request parsing
- `response_format` parsing
- consistent model IDs in `/v1/models`
- stable error JSON shape

## Current local streamed server status

Implemented:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- OpenAI-style JSON error envelope
- stable local model IDs in `/v1/models`
- non-stream chat responses with `usage`, `service_tier`, and `system_fingerprint`
- SSE streaming with `chat.completion.chunk`
- `stream_options.include_usage`
- simple stop-sequence trimming
- typed text-part normalization for input messages
- explicit rejection of unsupported `tools` and `response_format`

Missing or incomplete:

- `POST /v1/responses`
- actual `response_format` execution
- JSON mode execution
- structured outputs with `json_schema`
- actual request `tools`
- actual request `tool_choice`
- actual response `tool_calls`
- streamed tool-call deltas
- `tool` role round-trips
- content-part parity beyond simple text normalization
- `/v1/completions`
- client-oriented request/session surfaces beyond Chat Completions

## Practical implementation order

1. Stabilize chat-completions response and error shapes
2. Add strict request parsing for `response_format`
3. Add JSON mode
4. Add schema-based structured outputs
5. Add tool calling round-trip
6. Add streaming usage chunks and better SSE parity
7. Add `/v1/responses`

## Official references

- Chat Completions API reference:
  - https://platform.openai.com/docs/api-reference/chat/create-chat-completion
- Chat streaming reference:
  - https://platform.openai.com/docs/api-reference/chat-streaming
- Function calling guide:
  - https://platform.openai.com/docs/guides/function-calling
- Models list API reference:
  - https://platform.openai.com/docs/api-reference/models/list
- Error codes guide:
  - https://platform.openai.com/docs/guides/error-codes/api-error
