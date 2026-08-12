#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERIAL_PORT="${1:-/dev/ttyUSB0}"
VOLUME="${2:-0.2}"
ALSA_DEVICE="${3:-}"

if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  echo "Python environment not found. Run ./setup_raspberry_pi.sh first." >&2
  exit 1
fi

if [[ ! -e "$SERIAL_PORT" ]]; then
  echo "Serial device not found: $SERIAL_PORT" >&2
  echo "Check with: ls -l /dev/ttyUSB* /dev/ttyACM*" >&2
  exit 1
fi

COMMAND=(
  "$PROJECT_DIR/.venv/bin/python"
  "$PROJECT_DIR/tools/raspi_pcm16_stream.py"
  --port "$SERIAL_PORT"
  --speaker mix
  --volume "$VOLUME"
)

if [[ -n "$ALSA_DEVICE" ]]; then
  COMMAND+=(--speaker-device "$ALSA_DEVICE")
fi

exec "${COMMAND[@]}"
