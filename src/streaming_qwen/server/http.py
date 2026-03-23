from __future__ import annotations

import argparse
import copy
import json
import logging
import socket
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from .protocol import (
    ChatResponseBuilder,
    RequestParser,
    RequestValidationError,
    ServerCapabilities,
    UnsupportedFeatureError,
    build_request_id,
    error_payload,
    now_s,
    parse_tool_calls,
    prompt_tokens_from_messages,
    trim_stop,
    usage_payload,
    visible_text,
)
from .runtime_adapter import StreamedModelSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streamed Qwen chat server")
    parser.add_argument("--model", required=True, help="Local model snapshot path")
    parser.add_argument("--index", required=True, help="Path to expert index JSON")
    parser.add_argument(
        "--served-model-id",
        default=None,
        help="Model identifier to expose on the OpenAI-compatible API",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=9002, help="Bind port")
    parser.add_argument("--routed-top-k", type=int, default=4, help="MoE routed experts")
    parser.add_argument(
        "--prefill-top-k",
        type=int,
        default=None,
        help="MoE routed experts to use during prompt prefill; defaults to routed-top-k",
    )
    parser.add_argument("--max-tokens", type=int, default=2048, help="Default max tokens")
    parser.add_argument("--temp", type=float, default=0.7, help="Default sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Default sampling top-p")
    parser.add_argument(
        "--visible-stall-tokens",
        type=int,
        default=12,
        help="End buffered streaming once visible output has stopped changing for this many generated tokens",
    )
    parser.add_argument(
        "--native-reader",
        default=None,
        help="Path to native expert reader dylib",
    )
    parser.add_argument(
        "--component-workers",
        type=int,
        default=3,
        help="Concurrent component workers for the expert reader",
    )
    parser.add_argument(
        "--enable-prefetch",
        action="store_true",
        default=False,
        help="Enable same-layer speculative expert prefetch across generated tokens",
    )
    parser.add_argument(
        "--disable-prefetch",
        action="store_false",
        dest="enable_prefetch",
        help="Disable same-layer speculative expert prefetch",
    )
    parser.add_argument(
        "--moe-impl",
        choices=["streamed", "pipelined"],
        default="streamed",
        help="Routed-expert implementation to use",
    )
    parser.add_argument(
        "--fused-gate-up",
        action="store_true",
        help="Use fused gate/up/swiglu expert execution path",
    )
    parser.add_argument(
        "--compile-fused-gate-up",
        action="store_true",
        help="Use mx.compile for fused gate/up/swiglu when enabled",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=512,
        help="Prompt prefill step size for streamed generation",
    )
    parser.add_argument(
        "--prompt-cache-size",
        type=int,
        default=8,
        help="Maximum number of KV cache entries to keep in memory",
    )
    parser.add_argument(
        "--prompt-cache-bytes",
        default="1G",
        help="Maximum in-memory KV cache budget, e.g. 512M or 1G",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=8,
        help="Warmup generation length before serving",
    )
    parser.add_argument(
        "--warmup-prompt",
        default="Hello",
        help="Warmup prompt text",
    )
    parser.add_argument(
        "--kv-cache-dir",
        default=".run/kv-cache",
        help="Directory for persistent KV cache (safetensors). Empty string to disable.",
    )
    parser.add_argument(
        "--kv-cache-disk-bytes",
        default="0",
        help="Disk KV cache budget, e.g. 5G. 0 = 5× --prompt-cache-bytes.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=True,
        help="Enable the model chat template's thinking mode",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_false",
        dest="enable_thinking",
        help="Disable the model chat template's thinking mode",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


class StreamedAPIHandler(BaseHTTPRequestHandler):
    server_version = "streamed-qwen/0.2"
    protocol_version = "HTTP/1.1"

    def _session(self) -> StreamedModelSession:
        return self.server.session  # type: ignore[attr-defined]

    def _args(self) -> argparse.Namespace:
        return self.server.args  # type: ignore[attr-defined]

    def _parser(self) -> RequestParser:
        return self.server.request_parser  # type: ignore[attr-defined]

    def _responses(self) -> ChatResponseBuilder:
        return self.server.response_builder  # type: ignore[attr-defined]

    def _request_id(self) -> str:
        return getattr(self, "_request_id_value", f"req_{uuid.uuid4().hex}")

    def _set_request_id(self, request_id: str) -> None:
        self._request_id_value = request_id

    def _send_json(self, payload: dict[str, Any], *, status_code: int = 200, processing_ms: int | None = None) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("x-request-id", self._request_id())
        self.send_header("openai-version", "2020-10-01")
        if processing_ms is not None:
            self.send_header("openai-processing-ms", str(processing_ms))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _send_error(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        self._send_json(
            error_payload(message, error_type=error_type, param=param, code=code),
            status_code=status_code,
        )

    def _write_sse_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("x-request-id", self._request_id())
        self.send_header("openai-version", "2020-10-01")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def _write_sse_event(self, payload: dict[str, Any]) -> bool:
        try:
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            logging.info("client disconnected during stream")
            self.close_connection = True
            return False

    def _write_sse_done(self) -> None:
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            logging.info("client disconnected before [DONE]")
            self.close_connection = True

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        if self.path.startswith("/v1/models"):
            session = self._session()
            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": session.model_id,
                            "object": "model",
                            "created": now_s(),
                            "owned_by": "local",
                        }
                    ],
                }
            )
            return
        self._send_error("Not Found", status_code=404, error_type="not_found_error")

    def do_POST(self) -> None:
        self._set_request_id(f"req_{uuid.uuid4().hex}")
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self._send_error("Not Found", status_code=404, error_type="not_found_error")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(content_length).decode() or "{}")
        except Exception as exc:
            self._send_error(f"Invalid request body: {exc}", status_code=400)
            return

        logging.debug(
            "chat request path=%s request_id=%s headers=%s body=%s",
            self.path,
            self._request_id(),
            {
                key: self.headers.get(key)
                for key in ("Content-Type", "Authorization", "User-Agent")
                if self.headers.get(key) is not None
            },
            json.dumps(body, ensure_ascii=True)[:4000],
        )

        started_at = time.perf_counter()
        try:
            request = self._parser().parse_chat_request(body)
            if request.stream:
                self._handle_streaming_chat(request)
            else:
                self._handle_chat(request, started_at=started_at)
        except RequestValidationError as exc:
            self._send_error(str(exc), status_code=400)
        except UnsupportedFeatureError as exc:
            self._send_error(
                str(exc),
                status_code=400,
                code="unsupported_feature",
            )
        except (BrokenPipeError, ConnectionResetError):
            logging.info("client disconnected before response completed")
        except Exception as exc:
            logging.exception("chat request failed")
            if not self.wfile.closed:
                self._send_error(str(exc), status_code=500, error_type="server_error")

    def _make_sampler(self, request) -> Any:
        return make_sampler(
            temp=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
        )

    def _prefill_prompt_prefix(
        self,
        *,
        tokens: list[int],
        cache: Any,
        step_size: int,
        model: Any,
        session: StreamedModelSession,
        top_k: int,
    ) -> None:
        if not tokens:
            return
        session.set_top_k(top_k)
        processed = 0
        while processed < len(tokens):
            n_to_process = min(step_size, len(tokens) - processed)
            chunk = mx.array(tokens[processed : processed + n_to_process], mx.uint32)
            model(chunk[None], cache=cache)
            mx.eval([c.state for c in cache])
            processed += n_to_process
            mx.clear_cache()

    def _generate(self, request):
        session = self._session()
        # Model ID suffix "-nothink" disables thinking for that request,
        # allowing opencode to select fast-no-think mode via model name.
        enable_thinking = self._args().enable_thinking
        if request.model.endswith("-nothink"):
            enable_thinking = False
        prompt_tokens = prompt_tokens_from_messages(
            session.tokenizer,
            request.messages,
            enable_thinking=enable_thinking,
            chat_template_kwargs=request.chat_template_kwargs,
            tools=request.tools,
        )
        conversation_tokens = prompt_tokens_from_messages(
            session.tokenizer,
            request.messages,
            enable_thinking=enable_thinking,
            add_generation_prompt=False,
            chat_template_kwargs=request.chat_template_kwargs,
            tools=request.tools,
        )
        sampler = self._make_sampler(request)
        with session.lock:
            session.prompt_cache.log_cache_stats()
            cache, rest = session.prompt_cache.fetch_nearest_cache(
                session.cache_namespace, prompt_tokens
            )
            cached_tokens = len(prompt_tokens) - len(rest)
            cache_key = prompt_tokens[:]
            if cache is None:
                cache = make_prompt_cache(session.model)
            prefilled_tokens = 0

            # --- System-prefix checkpoint ---
            # Tokenize only the system message(s) (plus tools, which the Qwen3.5
            # template appends to the system block).  This prefix is identical
            # across every session that uses the same agent configuration, so
            # caching it lets fetch_nearest_cache return ~4 k cached tokens even
            # when the user message differs between sessions.
            system_messages = [m for m in request.messages if m["role"] == "system"]
            if system_messages:
                try:
                    # Qwen3.5's template requires a user message; pass two dummy
                    # variants and find the common prefix — that prefix is exactly
                    # the system block (including tool definitions appended by the
                    # template).  Any request with the same system config shares
                    # this prefix regardless of the actual user message.
                    _dummy_base = dict(chat_template_kwargs=request.chat_template_kwargs,
                                       tools=request.tools,
                                       enable_thinking=enable_thinking,
                                       add_generation_prompt=False)
                    _d1 = system_messages + [{"role": "user", "content": ""}]
                    _d2 = system_messages + [{"role": "user", "content": "."}]
                    _t1 = prompt_tokens_from_messages(session.tokenizer, _d1, **_dummy_base)
                    _t2 = prompt_tokens_from_messages(session.tokenizer, _d2, **_dummy_base)
                    sys_len = next(
                        (i for i, (a, b) in enumerate(zip(_t1, _t2)) if a != b),
                        min(len(_t1), len(_t2)),
                    )
                    system_tokens = _t1[:sys_len]
                except Exception:
                    system_tokens = []  # skip if template behaves unexpectedly
                    sys_len = 0
                else:
                    sys_len = len(system_tokens)
                if (
                    cached_tokens < sys_len < len(conversation_tokens)
                    and prompt_tokens[:sys_len] == system_tokens
                ):
                    sys_gap = sys_len - cached_tokens
                    sys_prefix = rest[:sys_gap]
                    if sys_prefix:
                        self._prefill_prompt_prefix(
                            tokens=sys_prefix,
                            cache=cache,
                            step_size=self._args().prefill_step_size,
                            model=session.model,
                            session=session,
                            top_k=session.prefill_top_k,
                        )
                        session.prompt_cache.insert_cache(
                            session.cache_namespace,
                            list(system_tokens),
                            copy.deepcopy(cache),
                            checkpoint=True,
                        )
                        rest = rest[sys_gap:]
                        prefilled_tokens = sys_gap

            # --- Conversation-prefix checkpoint (system + all messages except gen prompt) ---
            reusable_checkpoint_len = len(conversation_tokens)
            if (
                cached_tokens + prefilled_tokens < reusable_checkpoint_len < len(prompt_tokens)
                and prompt_tokens[:reusable_checkpoint_len] == conversation_tokens
            ):
                gap = reusable_checkpoint_len - (cached_tokens + prefilled_tokens)
                prefix = rest[:gap]
                if prefix:
                    self._prefill_prompt_prefix(
                        tokens=prefix,
                        cache=cache,
                        step_size=self._args().prefill_step_size,
                        model=session.model,
                        session=session,
                        top_k=session.prefill_top_k,
                    )
                    session.prompt_cache.insert_cache(
                        session.cache_namespace,
                        conversation_tokens,
                        copy.deepcopy(cache),
                        checkpoint=True,
                    )
                    rest = rest[gap:]
                    prefilled_tokens += gap
            prompt_checkpoint_saved = False

            if len(rest) > 1:
                prefill_tokens = rest[:-1]
                self._prefill_prompt_prefix(
                    tokens=prefill_tokens,
                    cache=cache,
                    step_size=self._args().prefill_step_size,
                    model=session.model,
                    session=session,
                    top_k=session.prefill_top_k,
                )
                rest = rest[-1:]
                prefilled_tokens += len(prefill_tokens)
                if cached_tokens + prefilled_tokens == len(prompt_tokens) - 1:
                    session.prompt_cache.insert_cache(
                        session.cache_namespace,
                        prompt_tokens[: cached_tokens + prefilled_tokens],
                        copy.deepcopy(cache),
                        checkpoint=True,
                    )
                    prompt_checkpoint_saved = True

            def save_prompt_checkpoint(processed_tokens: int, total_tokens: int) -> None:
                nonlocal prompt_checkpoint_saved
                if prompt_checkpoint_saved:
                    return
                checkpoint_len = cached_tokens + prefilled_tokens + processed_tokens
                if checkpoint_len != len(prompt_tokens) - 1:
                    return
                session.prompt_cache.insert_cache(
                    session.cache_namespace,
                    prompt_tokens[:checkpoint_len],
                    copy.deepcopy(cache),
                    checkpoint=True,
                )
                prompt_checkpoint_saved = True

            mx.reset_peak_memory()
            last_response = None
            try:
                session.set_top_k(session.decode_top_k)
                for response in stream_generate(
                    model=session.model,
                    tokenizer=session.tokenizer,
                    prompt=rest,
                    max_tokens=request.max_tokens,
                    sampler=sampler,
                    prompt_cache=cache,
                    prefill_step_size=self._args().prefill_step_size,
                    prompt_progress_callback=save_prompt_checkpoint,
                ):
                    cache_key.append(int(response.token))
                    last_response = response
                    yield response
            finally:
                session.set_top_k(session.decode_top_k)
                if last_response is not None:
                    logging.info(
                        "generation stats: prompt=%d cached=%d prefill=%.1f tok/s decode=%d tokens @ %.1f tok/s",
                        len(prompt_tokens),
                        cached_tokens,
                        getattr(last_response, "prompt_tps", 0),
                        getattr(last_response, "generation_tokens", 0),
                        getattr(last_response, "generation_tps", 0),
                    )
                # Store history-compatible checkpoint BEFORE insert_cache(cache_key, cache).
                # insert_cache with checkpoint=False can evict the oldest checkpoint (e.g.
                # system+tools) from memory.  _save_end_of_turn_checkpoint needs that checkpoint
                # as a base for re-prefill — if it's already evicted we re-prefill from scratch
                # (7530 tokens, 150s) instead of from the existing base (~2000 tokens, ~40s).
                self._save_end_of_turn_checkpoint(
                    session=session,
                    request=request,
                    prompt_tokens=prompt_tokens,
                    cache_key=cache_key,
                    enable_thinking=enable_thinking,
                    prefill_step_size=self._args().prefill_step_size,
                )
                # Store the full generation cache (may evict oldest checkpoint).
                session.prompt_cache.insert_cache(session.cache_namespace, cache_key, cache)
                # Flush queued checkpoint saves NOW — inference is done, Metal is idle.
                # save_prompt_cache issues Metal GPU→CPU blits; running it here (on the
                # handler thread, after stream_generate) avoids concurrent Metal access.
                session.prompt_cache.flush_pending_saves()
                session.prompt_cache.log_cache_stats()
        return {
            "prompt_tokens": len(prompt_tokens),
            "cached_tokens": cached_tokens,
        }

    def _save_end_of_turn_checkpoint(
        self,
        *,
        session: StreamedModelSession,
        request: Any,
        prompt_tokens: list[int],
        cache_key: list[int],
        enable_thinking: bool,
        prefill_step_size: int,
    ) -> None:
        """Store a KV cache checkpoint keyed to how this turn appears in future history.

        When enable_thinking=False, Qwen3's template injects <think>\\n\\n</think>\\n\\n tokens
        into the generation prompt.  These appear in cache_key but NOT in how the assistant
        turn is encoded in subsequent conversation history (template line 103 omits them for
        non-final turns).  Without this fix, turn N+1 can only reuse the cache up to the
        system+user block; the full assistant response must be re-prefilled every turn.

        Fix: decode the generated tokens, re-tokenize the completed turn as it will appear
        in history (no think injection), fetch the nearest existing checkpoint as a base,
        prefill the gap, and store a new checkpoint at the history-compatible key.
        """
        generated_ids = cache_key[len(prompt_tokens):]
        if not generated_ids:
            return
        assistant_text = session.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if not assistant_text.strip():
            return
        history_messages = list(request.messages) + [{"role": "assistant", "content": assistant_text}]
        try:
            # Append two dummy user variants so the template treats the assistant as a
            # history turn (loop.index0 <= last_query_index) rather than the final turn.
            # Without a subsequent user message, Qwen3's template injects
            # <think>\n\n</think>\n\n into the assistant block even with enable_thinking=False,
            # matching the generation-prompt injection but NOT what future turns encode.
            _dummy_base = dict(
                enable_thinking=enable_thinking,
                add_generation_prompt=False,
                chat_template_kwargs=request.chat_template_kwargs,
                tools=request.tools,
            )
            _d1 = prompt_tokens_from_messages(
                session.tokenizer, history_messages + [{"role": "user", "content": ""}], **_dummy_base
            )
            _d2 = prompt_tokens_from_messages(
                session.tokenizer, history_messages + [{"role": "user", "content": "."}], **_dummy_base
            )
            D = next(
                (i for i, (a, b) in enumerate(zip(_d1, _d2)) if a != b),
                min(len(_d1), len(_d2)),
            )
            # Strip <|im_start|>user\n (3 tokens) before divergence to isolate the
            # history-compatible assistant block end (through <|im_end|>\n).
            if D < 3:
                return
            end_of_turn_tokens = _d1[:D - 3]
        except Exception:
            return

        eot_cache, eot_rest = session.prompt_cache.fetch_nearest_cache(
            session.cache_namespace, end_of_turn_tokens
        )
        if not eot_rest:
            return  # already cached exactly
        if eot_cache is None:
            eot_cache = make_prompt_cache(session.model)
        self._prefill_prompt_prefix(
            tokens=eot_rest,
            cache=eot_cache,
            step_size=prefill_step_size,
            model=session.model,
            session=session,
            top_k=session.prefill_top_k,
        )
        session.prompt_cache.insert_cache(
            session.cache_namespace,
            end_of_turn_tokens,
            copy.deepcopy(eot_cache),
            checkpoint=True,
        )
        logging.info(
            "end-of-turn checkpoint stored: %d tokens (re-prefilled %d without think injection)",
            len(end_of_turn_tokens),
            len(eot_rest),
        )

    def _handle_chat(self, request, *, started_at: float) -> None:
        request_id = build_request_id("chatcmpl")
        created = now_s()
        full_text = ""
        emitted_text = ""
        final_response = None
        gen = self._generate(request)
        generation_meta = {"prompt_tokens": 0, "cached_tokens": 0}
        while True:
            try:
                response = next(gen)
            except StopIteration as stop:
                if stop.value:
                    generation_meta = stop.value
                break
            final_response = response
            full_text += response.text
            emitted_text, stop_hit = trim_stop(visible_text(full_text), request.stop_words)
            if stop_hit:
                break
        if final_response is None:
            raise RuntimeError("Generation produced no response")
        tool_calls = parse_tool_calls(full_text) if request.tools else None
        if tool_calls:
            finish_reason = "tool_calls"
            content = None
        else:
            finish_reason = "stop" if emitted_text != full_text else final_response.finish_reason
            content = emitted_text
        usage = usage_payload(
            final_response,
            prompt_tokens=generation_meta["prompt_tokens"],
            cached_tokens=generation_meta["cached_tokens"],
        )
        payload = self._responses().chat_completion(
            request_id=request_id,
            created=created,
            model=request.model,
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tool_calls,
        )
        self._send_json(
            payload,
            status_code=200,
            processing_ms=int((time.perf_counter() - started_at) * 1000),
        )

    def _handle_streaming_chat(self, request) -> None:
        request_id = build_request_id("chatcmpl")
        created = now_s()
        full_text = ""
        emitted_text = ""
        final_response = None
        generation_meta = {"prompt_tokens": 0, "cached_tokens": 0}
        stalled_visible_tokens = 0
        role_sent = False
        self._write_sse_headers()
        gen = self._generate(request)
        while True:
            try:
                response = next(gen)
            except StopIteration as stop:
                if stop.value:
                    generation_meta = stop.value
                break
            final_response = response
            full_text += response.text
            new_visible, stop_hit = trim_stop(visible_text(full_text), request.stop_words)
            if new_visible == emitted_text:
                stalled_visible_tokens += 1
            else:
                delta = new_visible[len(emitted_text) :]
                chunk = self._responses().stream_chunk(
                    request_id=request_id,
                    created=created,
                    model=request.model,
                    delta=(
                        {"role": "assistant", "content": delta or ""}
                        if not role_sent
                        else {"content": delta}
                    ),
                    finish_reason=None,
                    include_usage=request.include_usage,
                )
                if not self._write_sse_event(chunk):
                    return
                role_sent = True
                emitted_text = new_visible
                stalled_visible_tokens = 0
            if stop_hit:
                break
            # Don't apply stall limit when tools are active: tool call XML is invisible
            # by design and must complete before we can parse and emit it.
            stall_limit = self._args().visible_stall_tokens if not request.tools else 2048
            if emitted_text and stalled_visible_tokens >= stall_limit:
                logging.info(
                    "Ending buffered stream after %s hidden/stalled tokens",
                    stalled_visible_tokens,
                )
                break

        if final_response is None:
            raise RuntimeError("Generation produced no response")
        tool_calls = parse_tool_calls(full_text) if request.tools else None
        if tool_calls:
            # Emit role opener if not yet sent
            if not role_sent:
                role_chunk = self._responses().stream_chunk(
                    request_id=request_id,
                    created=created,
                    model=request.model,
                    delta={"role": "assistant", "content": None},
                    finish_reason=None,
                    include_usage=request.include_usage,
                )
                if not self._write_sse_event(role_chunk):
                    return
            # Emit one chunk per tool call
            for i, tc in enumerate(tool_calls):
                tc_chunk = self._responses().stream_chunk(
                    request_id=request_id,
                    created=created,
                    model=request.model,
                    delta={"tool_calls": [{
                        "index": i,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }]},
                    finish_reason=None,
                    include_usage=request.include_usage,
                )
                if not self._write_sse_event(tc_chunk):
                    return
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop" if emitted_text != full_text else final_response.finish_reason
            if not role_sent:
                empty_chunk = self._responses().stream_chunk(
                    request_id=request_id,
                    created=created,
                    model=request.model,
                    delta={"role": "assistant", "content": ""},
                    finish_reason=None,
                    include_usage=request.include_usage,
                )
                if not self._write_sse_event(empty_chunk):
                    return
        final_chunk = self._responses().stream_chunk(
            request_id=request_id,
            created=created,
            model=request.model,
            delta={},
            finish_reason=finish_reason,
            include_usage=request.include_usage,
        )
        if not self._write_sse_event(final_chunk):
            return
        if request.include_usage:
            usage_chunk = self._responses().final_stream_usage(
                request_id=request_id,
                created=created,
                model=request.model,
                usage=usage_payload(
                    final_response,
                    prompt_tokens=generation_meta["prompt_tokens"],
                    cached_tokens=generation_meta["cached_tokens"],
                ),
            )
            if not self._write_sse_event(usage_chunk):
                return
        self._write_sse_done()

    def log_message(self, format: str, *args) -> None:
        logging.info("%s - %s", self.address_string(), format % args)


def run_server(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    session = StreamedModelSession(args)
    session.warmup()
    server_address = (args.host, args.port)
    infos = socket.getaddrinfo(
        *server_address,
        type=socket.SOCK_STREAM,
        flags=socket.AI_PASSIVE,
    )
    ThreadingHTTPServer.address_family, _, _, _, bind_address = next(iter(infos))
    httpd = ThreadingHTTPServer(bind_address, StreamedAPIHandler)
    httpd.session = session  # type: ignore[attr-defined]
    httpd.args = args  # type: ignore[attr-defined]
    httpd.request_parser = RequestParser(  # type: ignore[attr-defined]
        default_model=session.model_id,
        default_max_tokens=args.max_tokens,
        default_temp=args.temp,
        default_top_p=args.top_p,
        capabilities=ServerCapabilities(
            tools=True,
            responses_api=False,
            json_mode=False,
            structured_outputs=False,
            logprobs=False,
        ),
    )
    httpd.response_builder = ChatResponseBuilder(  # type: ignore[attr-defined]
        model_id=session.model_id,
        system_fingerprint=session.system_fingerprint,
    )
    logging.info("Starting streamed Qwen server at http://%s:%s/v1", args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        session.close()
