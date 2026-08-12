#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/5] Installing Raspberry Pi system packages..."
sudo apt-get update
sudo apt-get install -y python3-venv alsa-utils

echo "[2/5] Creating Python virtual environment..."
python3 -m venv "$PROJECT_DIR/.venv"
"$PROJECT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$PROJECT_DIR/.venv/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"

echo "[3/5] Preparing output directory..."
mkdir -p "$PROJECT_DIR/recordings"

echo "[4/5] Checking serial access..."
if ! id -nG "$USER" | tr ' ' '\n' | grep -qx dialout; then
  sudo usermod -aG dialout "$USER"
  echo "Added $USER to dialout. Log out and reconnect before using the ESP32 serial port."
fi

echo "[5/5] Running receiver unit tests..."
"$PROJECT_DIR/.venv/bin/python" -m unittest "$PROJECT_DIR/tools/test_raspi_pcm16_stream.py" -v

echo
echo "Setup complete."
echo "Serial devices:"
find /dev -maxdepth 1 \( -name 'ttyUSB*' -o -name 'ttyACM*' \) -print 2>/dev/null || true
echo
echo "ALSA playback devices:"
aplay -l || true
echo
echo "After reconnecting, run:"
echo "  $PROJECT_DIR/run_speaker_bridge.sh /dev/ttyUSB0 0.2"
