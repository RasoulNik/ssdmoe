#!/usr/bin/env bash
set -euo pipefail

MODE="chat"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9002}"
MODEL="${MODEL:-streamed-qwen-k4}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TEMP="${TEMP:-0.2}"
TOP_P="${TOP_P:-0.95}"
PROMPT=""

usage() {
  cat <<EOF
Usage:
  ./scripts/streamed-chat.sh [--stream|--bench] [--model MODEL] [--max-tokens N] [--temp T] [--top-p P] [prompt...]

Defaults:
  HOST=$HOST
  PORT=$PORT
  MODEL=$MODEL
  MAX_TOKENS=$MAX_TOKENS
  TEMP=$TEMP
  TOP_P=$TOP_P

Examples:
  ./scripts/streamed-chat.sh "Say hello in one short sentence."
  ./scripts/streamed-chat.sh --stream "Say hello in one short sentence."
  ./scripts/streamed-chat.sh --bench "Count from 1 to 20, one number per line."
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stream)
      MODE="stream"
      shift
      ;;
    --bench)
      MODE="bench"
      shift
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --temp)
      TEMP="$2"
      shift 2
      ;;
    --top-p)
      TOP_P="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      PROMPT="${PROMPT:+$PROMPT }$1"
      shift
      ;;
  esac
done

PROMPT="${PROMPT:-Say hello in one short sentence.}"
URL="http://$HOST:$PORT/v1/chat/completions"

if [[ "$MODE" == "stream" ]]; then
  python3 - "$URL" "$MODEL" "$PROMPT" "$MAX_TOKENS" "$TEMP" "$TOP_P" <<'PY'
import json
import subprocess
import sys

url, model, prompt, max_tokens, temp, top_p = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(max_tokens),
    "temperature": float(temp),
    "top_p": float(top_p),
    "stream": True,
    "stream_options": {"include_usage": True},
}
subprocess.run(
    [
        "curl",
        "--no-buffer",
        "-sS",
        url,
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(payload),
    ],
    check=True,
)
PY
  exit 0
fi

if [[ "$MODE" == "bench" ]]; then
  python3 - "$URL" "$MODEL" "$PROMPT" "$MAX_TOKENS" "$TEMP" "$TOP_P" <<'PY'
import json
import subprocess
import sys
import time

url, model, prompt, max_tokens, temp, top_p = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(max_tokens),
    "temperature": float(temp),
    "top_p": float(top_p),
}
started = time.perf_counter()
proc = subprocess.run(
    [
        "curl",
        "-sS",
        url,
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(payload),
    ],
    check=True,
    capture_output=True,
    text=True,
)
elapsed = time.perf_counter() - started
response = json.loads(proc.stdout)
usage = response.get("usage", {})
completion_tokens = int(usage.get("completion_tokens", 0))
tps = (completion_tokens / elapsed) if elapsed > 0 else 0.0
content = response["choices"][0]["message"]["content"]

print("response:")
print(json.dumps(response, indent=2, ensure_ascii=False))
print()
print("metrics:")
print(f"elapsed_s={elapsed:.3f}")
print(f"completion_tokens={completion_tokens}")
print(f"tokens_per_second={tps:.2f}")
print(f"content={content!r}")
PY
  exit 0
fi

python3 - "$URL" "$MODEL" "$PROMPT" "$MAX_TOKENS" "$TEMP" "$TOP_P" <<'PY'
import json
import subprocess
import sys

url, model, prompt, max_tokens, temp, top_p = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(max_tokens),
    "temperature": float(temp),
    "top_p": float(top_p),
}
proc = subprocess.run(
    [
        "curl",
        "-sS",
        url,
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(payload),
    ],
    check=True,
    capture_output=True,
    text=True,
)
response = json.loads(proc.stdout)
print(json.dumps(response, indent=2, ensure_ascii=False))
PY
