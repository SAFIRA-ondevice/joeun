"""SAFIRA SPK0 packet codec.

Contract:
- mono S16LE
- 16 kHz
- 20 ms
- 320 samples / 640-byte PCM
- 8-byte <4sHH header
- 648-byte packet
"""
from __future__ import annotations

import struct
import numpy as np

MAGIC = b"SPK0"
SAMPLE_RATE = 16000
FRAME_SAMPLES = 320
PCM_BYTES = FRAME_SAMPLES * 2
HEADER = struct.Struct("<4sHH")
PACKET_BYTES = HEADER.size + PCM_BYTES


def pcm16_bytes(samples) -> bytes:
    x = np.asarray(samples)
    if x.size != FRAME_SAMPLES:
        raise ValueError(f"SPK0 requires exactly {FRAME_SAMPLES} samples, got {x.size}")
    if x.dtype != np.int16:
        x = np.clip(np.rint(x), -32768, 32767).astype(np.int16)
    return x.astype("<i2", copy=False).tobytes()


def pack_spk0(sequence: int, samples) -> bytes:
    payload = pcm16_bytes(samples)
    return HEADER.pack(MAGIC, int(sequence) & 0xFFFF, FRAME_SAMPLES) + payload


def unpack_spk0(packet: bytes) -> tuple[int, np.ndarray]:
    if len(packet) != PACKET_BYTES:
        raise ValueError(f"invalid SPK0 length={len(packet)}, expected={PACKET_BYTES}")
    magic, sequence, count = HEADER.unpack_from(packet)
    if magic != MAGIC:
        raise ValueError(f"invalid SPK0 magic={magic!r}")
    if count != FRAME_SAMPLES:
        raise ValueError(f"invalid SPK0 sample_count={count}")
    pcm = np.frombuffer(packet, dtype="<i2", count=FRAME_SAMPLES, offset=HEADER.size).copy()
    return sequence, pcm
