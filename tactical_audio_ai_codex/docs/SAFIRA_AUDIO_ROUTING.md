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

## Enrollment-free self-speech attribution

`self_speech.py` compares voice with each ambient side independently. It does
not learn a voiceprint, subtract a waveform, or implement acoustic echo
cancellation. Speaker-verification enrollment is a possible future optional
fallback, not a dependency or a currently implemented feature.

Both live scripts retain synchronized L/R/voice samples for the same trailing
1 s window used for the CNN. Every inference hop (default 250 ms, rounded up
to whole 20 ms packets), the detector:

1. Splits that window into 20 ms chunks and removes DC for measurement only.
2. Computes each side's maximum absolute normalized cross-correlation with
   voice over +/-32 samples (+/-2 ms). Absolute values tolerate inverted
   polarity; positive lag means ambient trails the reference.
3. Requires sufficient voice and ambient energy, a near-voice energy advantage,
   and sufficient correlation on the **same** ambient side. Either side may
   provide self leakage evidence. It does not average L/R before this test.
4. Combines the fraction of matching audible chunks with the unsmoothed voice
   AND ambient CNN speech scores from that exact window. Thus correlated
   non-speech alone does not establish self speech.
5. Smooths evidence and holds self state across short dropouts. Raw self
   candidates also block external attribution during the smoothing attack.
6. Marks external speech as a candidate only with ambient speech, audible
   ambient energy, insufficient reference correlation and no self gate.
   Correlated speech lacking near-voice evidence remains `uncertain`.

This is inference-cadence attribution, **not a 20 ms identity decision**.
The startup window is 1 s; trailing-window evidence, smoothing, inference time
and hangover add latency. Overlapping windows use newly captured sample time
for smoothing/hangover, not the full window length. A sequence discontinuity
clears windows, score history and detector state before rebuilding the window.
Between inferences the full-duplex loop uses the most recent state.

### Output and routing contract

Existing `targets`, `detected`, `final` and `speech_source=voice_channel` retain
their classifier meaning. They do not identify the speaker. New JSON fields:

| Field | Meaning |
|---|---|
| `self_speech_active` | Heuristic self leakage state, including hangover |
| `external_speech_candidate` | Eligible external-speech candidate after gating |
| `speech_attribution` | self_candidate / external_candidate / uncertain / no_ambient_speech |
| `ambient_speech_score`, `voice_speech_score` | Raw aligned CNN probabilities, 0..1 (unlike percent `targets`) |
| `self_speech_evidence` | Smoothed heuristic evidence, not a calibrated identity probability |
| `self_speech_raw_candidate` | Unsmooth self candidate; blocks premature external attribution |
| `self_speech_metrics` | Per-side correlation, lag, RMS, relative energy and vote fractions |
| `routing_detected` | Environmental labels plus eligible external_speech; never generic speech |
| `self_speech_pcm_removed` | Always false |
| `source_separation_available` | Always false in both live paths |

Correlation and dBFS metrics are averaged across chunks; lag is the median
best lag. These summaries are diagnostics, not additional classifier scores.
The router applies the same semantic gate even if passed raw `detected` labels.
Neither self state nor external eligibility changes existing gains. No playback
gain for external speech has been specified, so it does not enable playback.
Uplink still sends the dedicated voice channel at x1, including any acoustic
contamination captured there. Server voice and limiter processing are unchanged.

**Important:** with drone/gunshot in the mixed ambient stream, leaked self voice
still receives that stream's scalar gain. Gating removes self speech from
external-speech decisions only. Independent waveform gains require a genuine
separator; self-voice cancellation requires a separate adaptive acoustic stage.

### Configuration and tuning

Both scripts accept `--self-speech-config settings.json`; omit it for defaults.
Example overrides (PCM dBFS, not sound-pressure levels):

```json
{
  "max_lag_samples": 32,
  "correlation_threshold": 0.65,
  "voice_min_dbfs": -40.0,
  "ambient_min_dbfs": -55.0,
  "near_voice_ratio_db": 6.0,
  "evidence_fraction": 0.6,
  "smoothing_seconds": 0.25,
  "hangover_seconds": 0.5
}
```

These are uncalibrated starting defaults. Set microphone gains consistently;
measure distance, delay, quiet/noisy speech levels and leakage on the actual
headset. Tune ratio and correlation against false self/external assignments,
then smoothing/hangover against missed short words and switching latency.
CNN speech thresholds still come from the checkpoint or `--threshold`.
The sample rate/frame settings are fixed to 16000/320 for this implementation.

Reverberation, wind, clipping, AGC differences, shifted microphones, periodic
sounds and headset playback leakage can invalidate correlation/proximity cues.
A nearby external speaker can be correlated at all microphones; this may be
uncertain or falsely self. Concurrent wearer + external speech is not separated:
strong self evidence suppresses external candidacy for that entire window.
Low correlation also does not prove a different identity. Opposite-phase L/R
may cancel in the existing mono ambient CNN input even though this detector
measures the two sides separately. Therefore this is not reliable speaker ID.

### Validation status and follow-up

Run from the repository root:

```bash
python3 -m unittest discover -s tactical_audio_ai_codex/tests -v
```

Synthetic tests cover delayed/inverted leakage, independent speech, silence/DC,
correlated non-speech, ambiguous far-field audio, one-sided leakage, smoothing,
hangover, semantic gating, gains and SPK0. Mocked live-entrypoint tests verify
JSON, uplink/headset framing and sequence-gap resets without a checkpoint or
devices. They do not measure real-world accuracy or Raspberry Pi throughput.

Hardware validation remains required: wearer-only, external-only, both speaking,
drone/gunshot mixtures, movement, downlink playback, packet loss, and timing/load
with the actual checkpoint on Pi. Record synchronized 3-channel samples and
report false assignments and latency before relying on the heuristic.

Existing full-duplex prototype limitations remain: inference runs in the serial
loop, and the downlink reuses the most recently received frame when no new frame
arrives (no jitter buffer/expiry). Those transport/playback changes are separate
follow-ups; this patch preserves that path and makes no end-to-end claim.

## Continuous capture in the receive-only AI entry point

`live_m3c0_multilabel.py` now uses `audio_capture.ContinuousCapture`: one dedicated
serial reader runs independently of inference and console output. The consumer
takes the latest contiguous 50 packets (1 s) at each hop. Storage is bounded to
50 packets; there is no unbounded queue of inference jobs. `--torch-threads`
defaults to 1 to limit CPU contention. This is not a hard real-time guarantee:
Python scheduling, CPU/power effects and device/driver buffers still require Pi
measurements. A slow consumer can skip windows; this is not lossless recording.

The reader starts before START_INMP_ONLY is written and no longer sleeps for
200 ms with the stream unread. Serial has finite read/write timeouts; shutdown
cancels the read and joins the reader before closing the port. Receive/parser
errors propagate to the main thread instead of silently stopping capture.
This does not introduce STARTED acknowledgement or speaker-mode startup support.

The default runtime baud for this entry point is **1,000,000**, with explicit
override retained. Other legacy entry points are unchanged. The documented
3-channel payload needs 964,000 bit/s with 8N1, so 921600 is insufficient.

`audio_rx` reports cumulative packets, forward missing-frame estimates,
discontinuities, backward/restart events, latest jump, buffer occupancy/capacity
and high-water mark, overwritten unconsumed frames and skipped analysis hops.
Normal rolling-window replacement of already-used frames is not counted as an
unconsumed overwrite. With deliberately large hops, unconsumed overwrites can
also reflect the selected sampling cadence rather than overloaded inference.
Modulo-65536 forward gaps below half the sequence range are counted, including
large jumps; backward/restart cases are reported separately because the header
has no session identifier. Normal sequence wrap does not reset the window.

Any sequence discontinuity clears the capture window. After 50 contiguous
packets, the next consumer snapshot resets classifier history and attribution.
Results whose input was captured before a new discontinuity are tagged
`input_discontinuity_during_inference=true`; the receive-only script does not
use those results to play audio. Thread-safe snapshots never splice samples
across a known gap. Host frame ages and `analysis_window_age_ms` exclude device
capture time and time waiting in serial/USB buffers; they are not acoustic
end-to-end latency measurements.

Tests use the actual packet parser with fragmented fake serial reads. They
withhold the inference consumer while packets keep arriving, verify bounded
storage/latest-window selection and counters, and cover large gaps, wrap,
backward jumps, receive errors, invalid frames and blocked-read cancellation.
Mocked CLI tests check default/override baud and thread settings. No Pi click
removal has been verified. The full-duplex entry point is not converted by this
change and retains the limitations listed above.
