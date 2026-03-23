#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="${OPENCODE_SIMPLE_DIR:-$ROOT_DIR/.run/opencode-streamed-simple}"
CONFIG_SRC="$ROOT_DIR/configs/opencode-streamed-simple.json"
CONFIG_DST="$TMP_DIR/opencode.json"

mkdir -p "$TMP_DIR"
cp -f "$CONFIG_SRC" "$CONFIG_DST"
cd "$TMP_DIR"

if [[ "${1:-}" == "run" ]]; then
  shift
  AGENT="${OPENCODE_AGENT:-simple}"
  exec opencode run \
    --agent "$AGENT" \
    --model streamed/streamed-qwen-k4 \
    --title x \
    "$@"
fi

exec opencode "$@"
