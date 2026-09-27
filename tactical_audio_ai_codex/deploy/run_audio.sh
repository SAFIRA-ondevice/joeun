#!/bin/bash
set -euo pipefail
: "${SAFIRA_PYTHON:?Set SAFIRA_PYTHON in /etc/default/safira-audio}"
: "${SAFIRA_PORT:?Set SAFIRA_PORT}"
: "${SAFIRA_MODEL:?Set SAFIRA_MODEL}"
: "${SAFIRA_CLASSIFIER:?Set SAFIRA_CLASSIFIER}"
[[ -x "$SAFIRA_PYTHON" ]] || { echo "Python missing: $SAFIRA_PYTHON" >&2; exit 1; }
[[ -f "$SAFIRA_MODEL" && -f "$SAFIRA_CLASSIFIER" ]] || { echo "Model missing" >&2; exit 1; }
for ((n=0; n<30; n++)); do
    [[ -c "$SAFIRA_PORT" ]] && break
    sleep 1
done
[[ -c "$SAFIRA_PORT" ]] || { echo "Serial device missing: $SAFIRA_PORT" >&2; exit 1; }
extra=()
case "${SAFIRA_ATTACH:-0}" in
    0) ;;
    1) extra+=(--attach-speaker) ;;
    *) echo "SAFIRA_ATTACH must be 0 or 1" >&2; exit 1 ;;
esac
exec "$SAFIRA_PYTHON" -u live_all_mics_reenroll.py --model "$SAFIRA_MODEL" --classifier-model "$SAFIRA_CLASSIFIER" --port "$SAFIRA_PORT" --baud 1000000 --seconds 0 --speech-gain 1.5 --drone-gain 1.5 --gunshot-gain 0.15 --unknown-gain 1.5 --master 0.01 --combined-gain 1.0 "${extra[@]}"
