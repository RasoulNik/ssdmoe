#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"
PID_FILE="$RUN_DIR/mlx-lm.pid"
PORT_FILE="$RUN_DIR/mlx-lm.port"
HOST_FILE="$RUN_DIR/mlx-lm.host"
MODEL_FILE="$RUN_DIR/mlx-lm.model"
LOG_FILE="$RUN_DIR/mlx-lm.log"
SESSION_NAME="mlx-lm-server"

DEFAULT_MODEL_4B="mlx-community/Qwen3.5-4B-MLX-4bit"
DEFAULT_MODEL_9B="mlx-community/Qwen3.5-9B-MLX-4bit"
MODEL="${MODEL:-$DEFAULT_MODEL_4B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9000}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMP="${TEMP:-0.7}"
TOP_P="${TOP_P:-0.95}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"

usage() {
  cat <<EOF
Usage: ./scripts/mlx-server.sh <start|stop|status|restart> [4b|9b]

Environment overrides:
  MODEL=$DEFAULT_MODEL_4B
  HOST=127.0.0.1
  PORT=9000
  MAX_TOKENS=16384
  TEMP=0.7
  TOP_P=0.95
  ENABLE_THINKING=true
EOF
}

resolve_model() {
  local model_arg="${1:-}"
  case "$model_arg" in
    "" )
      ;;
    4b )
      MODEL="$DEFAULT_MODEL_4B"
      ;;
    9b )
      MODEL="$DEFAULT_MODEL_9B"
      ;;
    * )
      echo "unknown model preset: $model_arg"
      echo "use one of: 4b, 9b"
      exit 1
      ;;
  esac
}

port_in_use() {
  local port="$1"
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

server_pid() {
  if [[ -f "$PID_FILE" ]]; then
    cat "$PID_FILE"
    return 0
  fi
  return 1
}

server_port() {
  if [[ -f "$PORT_FILE" ]]; then
    cat "$PORT_FILE"
    return 0
  fi
  echo "$PORT"
}

server_host() {
  if [[ -f "$HOST_FILE" ]]; then
    cat "$HOST_FILE"
    return 0
  fi
  echo "$HOST"
}

server_model() {
  if [[ -f "$MODEL_FILE" ]]; then
    cat "$MODEL_FILE"
    return 0
  fi
  echo "$MODEL"
}

is_running() {
  local pid
  pid="$(server_pid)" || return 1
  kill -0 "$pid" 2>/dev/null
}

start_server() {
  mkdir -p "$RUN_DIR"

  if is_running; then
    local pid
    local running_host
    local running_port
    local running_model
    pid="$(server_pid)"
    running_host="$(server_host)"
    running_port="$(server_port)"
    running_model="$(server_model)"
    echo "mlx-lm server is already running"
    echo "pid: $pid"
    echo "model: $running_model"
    echo "endpoint: http://$running_host:$running_port/v1"
    echo "log: $LOG_FILE"
    return 0
  fi

  rm -f "$PID_FILE" "$PORT_FILE" "$HOST_FILE" "$MODEL_FILE"
  screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true

  cd "$ROOT_DIR"

  if port_in_use "$PORT"; then
    echo "port $PORT is already in use"
    return 1
  fi

  local chat_template_args=""
  if [[ "$ENABLE_THINKING" == "false" ]]; then
    chat_template_args="--chat-template-args '{\"enable_thinking\":false}'"
  fi

  screen -dmS "$SESSION_NAME" /bin/zsh -lc \
    "cd '$ROOT_DIR' && exec poetry run python -m mlx_lm server --model '$MODEL' --host '$HOST' --port '$PORT' --max-tokens '$MAX_TOKENS' --temp '$TEMP' --top-p '$TOP_P' $chat_template_args >>'$LOG_FILE' 2>&1"

  local pid=""
  local attempt=0
  while [[ "$attempt" -lt 30 ]]; do
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
    echo "$MODEL" > "$MODEL_FILE"
    echo "mlx-lm server started"
    echo "pid: $pid"
    echo "model: $MODEL"
    echo "endpoint: http://$HOST:$PORT/v1"
    echo "log: $LOG_FILE"
  else
    echo "mlx-lm server failed to start"
    echo "check log: $LOG_FILE"
    screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
    rm -f "$PID_FILE" "$PORT_FILE" "$HOST_FILE" "$MODEL_FILE"
    return 1
  fi
}

stop_server() {
  if ! [[ -f "$PID_FILE" ]]; then
    echo "mlx-lm server is not running"
    screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
    return 0
  fi

  local pid
  pid="$(server_pid)"

  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
    echo "stopped mlx-lm server"
    echo "pid: $pid"
  else
    echo "mlx-lm server is not running"
    echo "removed stale PID file"
    screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
  fi

  rm -f "$PID_FILE" "$PORT_FILE" "$HOST_FILE" "$MODEL_FILE"
}

status_server() {
  if is_running; then
    local pid
    local running_host
    local running_port
    local running_model
    pid="$(server_pid)"
    running_host="$(server_host)"
    running_port="$(server_port)"
    running_model="$(server_model)"
    echo "mlx-lm server is running"
    echo "pid: $pid"
    echo "model: $running_model"
    echo "endpoint: http://$running_host:$running_port/v1"
    echo "log: $LOG_FILE"
  else
    echo "mlx-lm server is not running"
    if [[ -f "$PID_FILE" ]]; then
      echo "stale PID file: $PID_FILE"
    fi
    return 1
  fi
}

ACTION="${1:-}"
MODEL_PRESET="${2:-}"

resolve_model "$MODEL_PRESET"

case "$ACTION" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  status)
    status_server
    ;;
  restart)
    stop_server
    start_server
    ;;
  *)
    usage
    exit 1
    ;;
esac
