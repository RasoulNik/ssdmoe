#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"
VENV_DIR="$ROOT_DIR/.open-webui-venv"
DATA_DIR="$ROOT_DIR/.open-webui-streamed-data"
PID_FILE="$RUN_DIR/open-webui-streamed.pid"
PORT_FILE="$RUN_DIR/open-webui-streamed.port"
HOST_FILE="$RUN_DIR/open-webui-streamed.host"
LOG_FILE="$RUN_DIR/open-webui-streamed.log"
SESSION_NAME="open-webui-streamed"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-3002}"
STREAMED_BASE_URL="${STREAMED_BASE_URL:-http://127.0.0.1:9002/v1}"
STREAMED_API_KEY="${STREAMED_API_KEY:-dummy}"
WEBUI_AUTH="${WEBUI_AUTH:-False}"
OFFLINE_MODE="${OFFLINE_MODE:-True}"

usage() {
  cat <<EOF
Usage: ./scripts/open-webui-streamed.sh <start|stop|status>

Environment overrides:
  HOST=$HOST
  PORT=$PORT
  STREAMED_BASE_URL=$STREAMED_BASE_URL
  STREAMED_API_KEY=$STREAMED_API_KEY
  WEBUI_AUTH=$WEBUI_AUTH
  OFFLINE_MODE=$OFFLINE_MODE
EOF
}

port_in_use() {
  local port="$1"
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

webui_pid() {
  [[ -f "$PID_FILE" ]] && cat "$PID_FILE"
}

webui_port() {
  [[ -f "$PORT_FILE" ]] && cat "$PORT_FILE" || echo "$PORT"
}

webui_host() {
  [[ -f "$HOST_FILE" ]] && cat "$HOST_FILE" || echo "$HOST"
}

is_running() {
  local pid
  pid="$(webui_pid)" || return 1
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

python_bin() {
  if command -v pyenv >/dev/null 2>&1; then
    pyenv which python
    return 0
  fi
  command -v python3
}

ensure_webui_installed() {
  if [[ -x "$VENV_DIR/bin/open-webui" ]]; then
    return 0
  fi
  mkdir -p "$RUN_DIR"
  local pybin
  pybin="$(python_bin)"
  "$pybin" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install open-webui
}

start_server() {
  mkdir -p "$RUN_DIR" "$DATA_DIR"

  if is_running; then
    echo "Open WebUI streamed is already running"
    echo "pid: $(webui_pid)"
    echo "url: http://$(webui_host):$(webui_port)"
    echo "backend: $STREAMED_BASE_URL"
    echo "log: $LOG_FILE"
    return 0
  fi

  "$ROOT_DIR/scripts/streamed-qwen-server.sh" start >/dev/null

  if port_in_use "$PORT"; then
    echo "port $PORT is already in use"
    return 1
  fi

  ensure_webui_installed
  rm -f "$PID_FILE" "$PORT_FILE" "$HOST_FILE"
  screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true

  screen -dmS "$SESSION_NAME" /bin/zsh -lc "
    cd '$ROOT_DIR' && \
    export DATA_DIR='$DATA_DIR' && \
    export WEBUI_AUTH='$WEBUI_AUTH' && \
    export OFFLINE_MODE='$OFFLINE_MODE' && \
    export ENABLE_OLLAMA_API='False' && \
    export ENABLE_OPENAI_API='True' && \
    export OPENAI_API_BASE_URL='$STREAMED_BASE_URL' && \
    export OPENAI_API_KEY='$STREAMED_API_KEY' && \
    exec '$VENV_DIR/bin/open-webui' serve --host '$HOST' --port '$PORT' >>'$LOG_FILE' 2>&1
  "

  local pid=""
  local attempt=0
  while [[ "$attempt" -lt 180 ]]; do
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
    echo "Open WebUI streamed started"
    echo "pid: $pid"
    echo "url: http://$HOST:$PORT"
    echo "backend: $STREAMED_BASE_URL"
    echo "log: $LOG_FILE"
  else
    echo "Open WebUI streamed failed to start"
    echo "check log: $LOG_FILE"
    screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
    rm -f "$PID_FILE" "$PORT_FILE" "$HOST_FILE"
    return 1
  fi
}

stop_server() {
  if ! [[ -f "$PID_FILE" ]]; then
    echo "Open WebUI streamed is not running"
    screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
    return 0
  fi
  local pid
  pid="$(webui_pid)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
    echo "stopped Open WebUI streamed"
    echo "pid: $pid"
  else
    echo "Open WebUI streamed is not running"
    echo "removed stale PID file"
    screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
  fi
  rm -f "$PID_FILE" "$PORT_FILE" "$HOST_FILE"
}

status_server() {
  if is_running; then
    echo "Open WebUI streamed is running"
    echo "pid: $(webui_pid)"
    echo "url: http://$(webui_host):$(webui_port)"
    echo "backend: $STREAMED_BASE_URL"
    echo "log: $LOG_FILE"
  else
    echo "Open WebUI streamed is not running"
    [[ -f "$PID_FILE" ]] && echo "stale PID file: $PID_FILE"
    return 1
  fi
}

ACTION="${1:-}"
case "$ACTION" in
  start) start_server ;;
  stop) stop_server ;;
  status) status_server ;;
  *) usage; exit 1 ;;
esac
