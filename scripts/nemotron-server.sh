#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"
PID_FILE="$RUN_DIR/nemotron.pid"
PORT_FILE="$RUN_DIR/nemotron.port"
HOST_FILE="$RUN_DIR/nemotron.host"
LOG_FILE="$RUN_DIR/nemotron.log"
SESSION_NAME="nemotron-server"
LOCK_FILE="$RUN_DIR/moe-server.lock"

MODEL_PATH="${MODEL_PATH:-$HOME/.cache/huggingface/hub/models--mlx-community--NVIDIA-Nemotron-3-Nano-30B-A3B-4bit/snapshots/832f602eba5d22436c258c1462bdedc5afddb42b}"
INDEX_PATH="${INDEX_PATH:-$ROOT_DIR/.run/nemotron30b-expert-index.json}"
NATIVE_READER_PATH="${NATIVE_READER_PATH:-$ROOT_DIR/.run/libexpert_reader.dylib}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9003}"
ROUTED_TOP_K="${ROUTED_TOP_K:-6}"
PREFILL_TOP_K="${PREFILL_TOP_K:-$ROUTED_TOP_K}"
COMPONENT_WORKERS="${COMPONENT_WORKERS:-3}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMP="${TEMP:-0.7}"
TOP_P="${TOP_P:-0.95}"
WARMUP_TOKENS="${WARMUP_TOKENS:-8}"
PREFILL_STEP_SIZE="${PREFILL_STEP_SIZE:-4096}"
PROMPT_CACHE_SIZE="${PROMPT_CACHE_SIZE:-8}"
PROMPT_CACHE_BYTES="${PROMPT_CACHE_BYTES:-1G}"
VISIBLE_STALL_TOKENS="${VISIBLE_STALL_TOKENS:-12}"
ENABLE_PREFETCH="${ENABLE_PREFETCH:-false}"

usage() {
  cat <<EOF
Usage: ./scripts/nemotron-server.sh <start|stop|status|restart>

Environment overrides:
  MODEL_PATH=$MODEL_PATH
  INDEX_PATH=$INDEX_PATH
  NATIVE_READER_PATH=$NATIVE_READER_PATH
  HOST=$HOST
  PORT=$PORT
  ROUTED_TOP_K=$ROUTED_TOP_K
  PREFILL_TOP_K=$PREFILL_TOP_K
  COMPONENT_WORKERS=$COMPONENT_WORKERS
  MAX_TOKENS=$MAX_TOKENS
  TEMP=$TEMP
  TOP_P=$TOP_P
  WARMUP_TOKENS=$WARMUP_TOKENS
  PREFILL_STEP_SIZE=$PREFILL_STEP_SIZE
  PROMPT_CACHE_SIZE=$PROMPT_CACHE_SIZE
  PROMPT_CACHE_BYTES=$PROMPT_CACHE_BYTES
  VISIBLE_STALL_TOKENS=$VISIBLE_STALL_TOKENS
  ENABLE_PREFETCH=$ENABLE_PREFETCH
EOF
}

port_in_use() {
  local port="$1"
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

server_pid() {
  [[ -f "$PID_FILE" ]] && cat "$PID_FILE"
}

server_port() {
  [[ -f "$PORT_FILE" ]] && cat "$PORT_FILE" || echo "$PORT"
}

server_host() {
  [[ -f "$HOST_FILE" ]] && cat "$HOST_FILE" || echo "$HOST"
}

is_running() {
  local pid
  pid="$(server_pid)" || return 1
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

validate_paths() {
  [[ -d "$MODEL_PATH" ]] || { echo "missing model snapshot: $MODEL_PATH"; exit 1; }
  [[ -f "$INDEX_PATH" ]] || { echo "missing expert index: $INDEX_PATH"; exit 1; }
  [[ -f "$NATIVE_READER_PATH" ]] || { echo "missing native reader dylib: $NATIVE_READER_PATH"; exit 1; }
}

start_server() {
  mkdir -p "$RUN_DIR"
  validate_paths

  if [[ -f "$LOCK_FILE" ]]; then
    local other
    other="$(cat "$LOCK_FILE")"
    echo "another MoE server is already running: $other"
    echo "stop it first to avoid OOM (only one model fits in memory)"
    return 1
  fi

  if is_running; then
    local pid
    pid="$(server_pid)"
    echo "Nemotron server is already running"
    echo "pid: $pid"
    echo "endpoint: http://$(server_host):$(server_port)/v1"
    echo "log: $LOG_FILE"
    return 0
  fi

  if port_in_use "$PORT"; then
    echo "port $PORT is already in use"
    return 1
  fi

  rm -f "$PID_FILE" "$PORT_FILE" "$HOST_FILE"
  screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true

  local prefetch
  if [[ "$ENABLE_PREFETCH" == "true" ]]; then
    prefetch="--enable-prefetch"
  else
    prefetch="--disable-prefetch"
  fi

  screen -dmS "$SESSION_NAME" /bin/zsh -lc "
    cd '$ROOT_DIR' && \
    export PYTHONPATH='$ROOT_DIR/src' && \
    exec poetry run python scripts/streamed_qwen_server.py \
      --model '$MODEL_PATH' \
      --index '$INDEX_PATH' \
      --host '$HOST' \
      --port '$PORT' \
      --routed-top-k '$ROUTED_TOP_K' \
      --prefill-top-k '$PREFILL_TOP_K' \
      --native-reader '$NATIVE_READER_PATH' \
      --component-workers '$COMPONENT_WORKERS' \
      --max-tokens '$MAX_TOKENS' \
      --temp '$TEMP' \
      --top-p '$TOP_P' \
      --visible-stall-tokens '$VISIBLE_STALL_TOKENS' \
      --prefill-step-size '$PREFILL_STEP_SIZE' \
      --prompt-cache-size '$PROMPT_CACHE_SIZE' \
      --prompt-cache-bytes '$PROMPT_CACHE_BYTES' \
      --warmup-tokens '$WARMUP_TOKENS' \
      $prefetch >>'$LOG_FILE' 2>&1
  "

  local pid=""
  local attempt=0
  while [[ "$attempt" -lt 120 ]]; do
    if port_in_use "$PORT"; then
      pid="$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN | head -n 1)"
      break
    fi
    sleep 1
    attempt=$((attempt + 1))
  done

  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "$pid" > "$PID_FILE"
    echo "$PORT" > "$PORT_FILE"
    echo "$HOST" > "$HOST_FILE"
    echo "nemotron (pid=$pid port=$PORT)" > "$LOCK_FILE"
    echo "Nemotron server started"
    echo "pid: $pid"
    echo "endpoint: http://$HOST:$PORT/v1"
    echo "routed_top_k: $ROUTED_TOP_K"
    echo "prefill_top_k: $PREFILL_TOP_K"
    echo "component_workers: $COMPONENT_WORKERS"
    echo "prompt_cache_size: $PROMPT_CACHE_SIZE"
    echo "prompt_cache_bytes: $PROMPT_CACHE_BYTES"
    echo "log: $LOG_FILE"
  else
    echo "Nemotron server failed to start"
    echo "check log: $LOG_FILE"
    screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
    rm -f "$PID_FILE" "$PORT_FILE" "$HOST_FILE"
    return 1
  fi
}

stop_server() {
  if ! [[ -f "$PID_FILE" ]]; then
    echo "Nemotron server is not running"
    screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
    return 0
  fi

  local pid
  pid="$(server_pid)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
    echo "stopped Nemotron server"
    echo "pid: $pid"
  else
    echo "Nemotron server is not running"
    echo "removed stale PID file"
    screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
  fi

  rm -f "$PID_FILE" "$PORT_FILE" "$HOST_FILE" "$LOCK_FILE"
}

status_server() {
  if is_running; then
    echo "Nemotron server is running"
    echo "pid: $(server_pid)"
    echo "endpoint: http://$(server_host):$(server_port)/v1"
    echo "log: $LOG_FILE"
  else
    echo "Nemotron server is not running"
    [[ -f "$PID_FILE" ]] && echo "stale PID file: $PID_FILE"
    return 1
  fi
}

ACTION="${1:-}"
case "$ACTION" in
  start) start_server ;;
  stop) stop_server ;;
  status) status_server ;;
  restart) stop_server; start_server ;;
  *) usage; exit 1 ;;
esac
