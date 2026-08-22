# M3C0 protocol

All integers and PCM samples are little-endian.

| Offset | Size | Meaning |
|---:|---:|---|
| 0 | 4 | ASCII `M3C0` |
| 4 | 2 | sequence, uint16 wrapping at 65535 |
| 6 | 2 | samples per channel, uint16 (normally 320) |
| 8 | `samples × 3 × 2` | interleaved signed PCM16 |

Payload order repeats `[ambient_left, ambient_right, voice]` for each sample. At 16 kHz, 320 samples are 20 ms and the payload is 1,920 bytes. The complete packet is 1,928 bytes.

The Pi normally sends `START_INMP_ONLY\n`; the ESP32 responds `STARTED\n` and begins binary packets. The parser resynchronizes by scanning for `M3C0` and warns on sequence gaps.
