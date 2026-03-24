from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ServerCapabilities:
    tools: bool = False
    responses_api: bool = False
    json_mode: bool = False
    structured_outputs: bool = False
    logprobs: bool = False


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: list[dict[str, Any]]
    stream: bool
    include_usage: bool
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    stop_words: list[str]
    chat_template_kwargs: dict[str, Any] | None
    response_format: dict[str, Any] | None
    session_id: str | None
    tools: list[dict[str, Any]] | None


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = [f.get("text", "") for f in content if f.get("type") == "text"]
            content = "".join(text_parts)
        elif content is None:
            content = ""
        norm: dict[str, Any] = {"role": role, "content": content}
        # Preserve tool_calls for assistant messages.
        # Qwen3.5 template iterates tool_call.function.arguments as a dict,
        # so parse any JSON-string arguments back to dicts.
        if role == "assistant" and message.get("tool_calls"):
            fixed_calls = []
            for tc in message["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                fixed_calls.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args},
                })
            norm["tool_calls"] = fixed_calls
        # tool_call_id is not used by the Qwen3.5 template (it just uses content)
        normalized.append(norm)
    return normalized


def prompt_from_messages(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool,
    chat_template_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    kwargs = dict(chat_template_kwargs or {})
    kwargs.setdefault("enable_thinking", enable_thinking)
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(
        normalize_messages(messages),
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def prompt_tokens_from_messages(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool,
    add_generation_prompt: bool = True,
    chat_template_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    kwargs = dict(chat_template_kwargs or {})
    kwargs.setdefault("enable_thinking", enable_thinking)
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(
        normalize_messages(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict[str, Any]] | None:
    """Parse Qwen3.5 XML-style tool calls from generated text.

    Format:
        <tool_call>
        <function=name>
        <parameter=param>value</parameter>
        </function>
        </tool_call>
    """
    tc_matches = _TOOL_CALL_RE.findall(text)
    if not tc_matches:
        return None
    calls = []
    for tc_content in tc_matches:
        fn_match = _FUNCTION_RE.search(tc_content)
        if not fn_match:
            continue
        fn_name = fn_match.group(1).strip()
        fn_body = fn_match.group(2)
        params: dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(fn_body):
            param_name = pm.group(1).strip()
            param_value = pm.group(2).strip()
            try:
                params[param_name] = json.loads(param_value)
            except json.JSONDecodeError:
                params[param_name] = param_value
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": fn_name,
                "arguments": json.dumps(params),
            },
        })
    return calls or None


def trim_stop(full_text: str, stop_words: list[str]) -> tuple[str, bool]:
    if not stop_words:
        return full_text, False
    trim_to = len(full_text)
    matched = False
    for stop_word in stop_words:
        if not stop_word:
            continue
        idx = full_text.find(stop_word)
        if idx != -1:
            trim_to = min(trim_to, idx)
            matched = True
    return full_text[:trim_to], matched


def visible_text(full_text: str) -> str:
    close_tag = "</think>"
    stripped = full_text.lstrip()
    if close_tag in full_text:
        text = full_text.split(close_tag, 1)[1].lstrip()
    elif stripped.startswith("<"):
        return ""
    else:
        text = full_text
    # Suppress tool_call blocks and everything after them
    if text.startswith("<"):
        return ""
    tc_idx = text.find("<tool_call>")
    return text[:tc_idx].rstrip() if tc_idx != -1 else text


def build_system_fingerprint(model_id: str, model_path: str, top_k: int) -> str:
    digest = hashlib.sha1(f"{model_id}:{model_path}:{top_k}".encode()).hexdigest()
    return f"fp_{digest[:10]}"


def build_request_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


def usage_payload(
    final_response: Any,
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cached_tokens: int = 0,
) -> dict[str, Any]:
    prompt_tokens = int(final_response.prompt_tokens if prompt_tokens is None else prompt_tokens)
    completion_tokens = int(
        final_response.generation_tokens if completion_tokens is None else completion_tokens
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {
            "cached_tokens": int(cached_tokens),
            "audio_tokens": 0,
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    }


def error_payload(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


class RequestValidationError(ValueError):
    pass


class UnsupportedFeatureError(NotImplementedError):
    pass


class RequestParser:
    def __init__(self, *, default_model: str, default_max_tokens: int, default_temp: float, default_top_p: float, capabilities: ServerCapabilities):
        self.default_model = default_model
        self.default_max_tokens = default_max_tokens
        self.default_temp = default_temp
        self.default_top_p = default_top_p
        self.capabilities = capabilities

    def _validate_response_format(
        self, response_format: Any, messages: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if response_format is None:
            return None
        if not isinstance(response_format, dict):
            raise RequestValidationError("response_format must be an object")
        response_type = response_format.get("type")
        if response_type not in {"json_object", "json_schema"}:
            raise RequestValidationError("unsupported response_format.type")
        if response_type == "json_schema" and not self.capabilities.structured_outputs:
            raise UnsupportedFeatureError("response_format json_schema is not implemented yet")
        if response_type == "json_object":
            if not self.capabilities.json_mode:
                raise UnsupportedFeatureError("response_format json_object is not implemented yet")
            normalized = normalize_messages(messages)
            if not any("JSON" in str(m.get("content", "")) for m in normalized):
                raise RequestValidationError(
                    'JSON mode requires the string "JSON" to appear somewhere in the conversation context'
                )
        return response_format

    def parse_chat_request(self, body: dict[str, Any]) -> ChatRequest:
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise RequestValidationError("messages must be a list")
        n = int(body.get("n", 1))
        if n != 1:
            raise RequestValidationError("only n=1 is supported")
        tools = body.get("tools") or None
        if tools and not self.capabilities.tools:
            raise UnsupportedFeatureError("tools are not implemented yet")
        if body.get("functions"):
            raise UnsupportedFeatureError("functions are not implemented yet")
        if body.get("tool_choice") not in (None, "none", "auto", "required"):
            raise UnsupportedFeatureError("tool_choice forcing is not implemented yet")
        response_format = self._validate_response_format(body.get("response_format"), messages)
        stop = body.get("stop") or []
        stop_words = [stop] if isinstance(stop, str) else list(stop)
        stream_options = body.get("stream_options") or {}
        if not isinstance(stream_options, dict):
            raise RequestValidationError("stream_options must be an object")
        requested_max_tokens = int(
            body.get("max_completion_tokens", body.get("max_tokens", self.default_max_tokens))
        )
        return ChatRequest(
            model=str(body.get("model") or self.default_model),
            messages=messages,
            stream=bool(body.get("stream", False)),
            include_usage=bool(stream_options.get("include_usage", False)),
            max_tokens=requested_max_tokens,
            temperature=float(body.get("temperature", self.default_temp)),
            top_p=float(body.get("top_p", self.default_top_p)),
            top_k=int(body.get("top_k", 0)),
            min_p=float(body.get("min_p", 0.0)),
            stop_words=stop_words,
            chat_template_kwargs=body.get("chat_template_kwargs"),
            response_format=response_format,
            session_id=(str(body["session_id"]) if body.get("session_id") else None),
            tools=tools,
        )


class ChatResponseBuilder:
    def __init__(self, *, model_id: str, system_fingerprint: str, service_tier: str = "default"):
        self.model_id = model_id
        self.system_fingerprint = system_fingerprint
        self.service_tier = service_tier

    def chat_completion(
        self,
        *,
        request_id: str,
        created: int,
        model: str,
        content: str | None,
        finish_reason: str | None,
        usage: dict[str, Any],
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "refusal": None,
                        "annotations": [],
                        "tool_calls": tool_calls,
                        "function_call": None,
                    },
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
            "service_tier": self.service_tier,
            "system_fingerprint": self.system_fingerprint,
        }

    def stream_chunk(
        self,
        *,
        request_id: str,
        created: int,
        model: str,
        delta: dict[str, Any],
        finish_reason: str | None,
        include_usage: bool,
    ) -> dict[str, Any]:
        payload = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": self.system_fingerprint,
            "service_tier": self.service_tier,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if include_usage:
            payload["usage"] = None
        return payload

    def final_stream_usage(
        self,
        *,
        request_id: str,
        created: int,
        model: str,
        usage: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": self.system_fingerprint,
            "service_tier": self.service_tier,
            "choices": [],
            "usage": usage,
        }


def now_s() -> int:
    return int(time.time())
