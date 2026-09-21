# SAFIRA audio routing contract

## Current transport contract

- ESP32 input: `M3C0`
- 16,000 Hz / signed PCM16 little-endian
- 3 channels: `ambient_left`, `ambient_right`, `voice`
- 20 ms / 320 samples per channel
- AI/DSP output: mono S16LE, 320 samples = 640 bytes
- network packet: 8-byte `SPK0` header + 640-byte PCM = 648-byte UDP datagram
- WireGuard carries the UDP traffic; AWS EC2 is not the AI/DSP processor.

## Gain and destination policy

| Source | Gain | Destination |
|---|---:|---|
| separated gunshot | x0.15 | local headset |
| separated drone | x1.50 | local headset |
| user voice | x1.00 | server uplink |
| unknown | x0.00 | local headset |
| server voice | x1.00 | local headset |

The local headset mix is peak-limited after summing streams to avoid int16
clipping, especially when the x1.50 drone stream and server voice overlap.

## Current model limitation

The current CNN is a multi-label **classifier**, not a source-separation model.
A probability such as `drone=0.8` is not playable drone-only PCM.

Therefore:

- `audio_routing.SafiraAudioRouter.headset_from_separated()` is reserved for
  real separated PCM produced by a future separator.
- `classification_guided_ambient()` is the current prototype fallback. It
  applies one scalar gain to the mixed ambient L/R waveform.
- If gunshot and drone are both detected in the same mixed waveform, gunshot
  attenuation has protective priority because independent per-source gain is
  impossible without source separation.
- The dedicated third microphone is used for the speech score and for user
  voice uplink. The old live script ignored this channel for inference.

## Full-duplex path

```text
M3C0 ambient L/R -> environmental classifier -> mixed fallback DSP ->+
                                                                   |
server SPK0 voice x1.0 --------------------------------------------+-> limiter -> headset

M3C0 voice channel -> speech classifier
                  \-> user voice x1.0 -> SPK0 UDP -> server
```

When a real source separator is available, replace the mixed fallback with:

```text
separator
  |- gunshot PCM x0.15 --+
  |- drone PCM   x1.50 --+-> mixer -> limiter -> headset
  |- unknown PCM x0.00 --+
server voice      x1.00 --+
```

Do not report the fallback stream as `separated`.

## Prototype command

The server address and ports are intentionally runtime parameters because the
repository does not define their final values.

```bash
python3 raspberry_pi/live_m3c0_full_duplex.py \
  --port /dev/ttyUSB1 \
  --baud 921600 \
  --model models/audio_cnn_multilabel_best.pt \
  --server-host <WIREGUARD_PEER_IP> \
  --uplink-port <UPLINK_PORT> \
  --downlink-port <DOWNLINK_PORT> \
  --expected-downlink-peer <WIREGUARD_PEER_IP> \
  --headset-serial
```

Before using `--headset-serial`, confirm the current ESP32 firmware accepts
`SPK0` playback packets while it is streaming `M3C0`.
