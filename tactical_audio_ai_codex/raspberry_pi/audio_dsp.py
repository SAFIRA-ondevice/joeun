"""DSP helpers for SAFIRA real-time audio routing.

Gain policy:
- separated gunshot: x0.15 -> local headset
- separated drone: x1.50 -> local headset
- user voice: x1.00 -> server uplink
- unknown: x0.00 -> local headset
- server voice: x1.00 -> local headset

This is digital PCM processing. dBFS is not acoustic SPL and this module is
not a certified hearing-protection implementation.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

GUNSHOT_GAIN = 0.15
DRONE_GAIN = 1.50
USER_VOICE_GAIN = 1.00
UNKNOWN_GAIN = 0.00
SERVER_VOICE_GAIN = 1.00


def linear_to_db(gain: float) -> float:
    if gain <= 0:
        return float("-inf")
    return 20.0 * math.log10(float(gain))


def peak_dbfs(samples: np.ndarray) -> float:
    x = np.asarray(samples)
    if x.size == 0:
        return -120.0
    peak = float(np.max(np.abs(x.astype(np.float32)))) / 32768.0
    return -120.0 if peak <= 1e-9 else 20.0 * math.log10(peak)


def apply_linear_gain(samples: np.ndarray, gain: float) -> np.ndarray:
    """Apply scalar gain and return clipped PCM16."""
    x = np.asarray(samples, dtype=np.float32) * float(gain)
    return np.clip(np.rint(x), -32768, 32767).astype(np.int16)


def mix_float_pcm16(streams: Iterable[tuple[np.ndarray, float]], frame_samples: int) -> np.ndarray:
    """Mix PCM16 streams in float domain without clipping until the final step."""
    mixed = np.zeros(frame_samples, dtype=np.float32)
    for samples, gain in streams:
        if samples is None:
            continue
        x = np.asarray(samples)
        if x.size != frame_samples:
            raise ValueError(f"expected {frame_samples} samples, got {x.size}")
        mixed += x.astype(np.float32) * float(gain)
    return mixed


class SmoothedGain:
    def __init__(self, initial: float = 1.0, attack_ms: float = 2.0, release_ms: float = 250.0):
        self.current = float(initial)
        self.attack_ms = max(float(attack_ms), 0.1)
        self.release_ms = max(float(release_ms), 0.1)

    def update(self, target: float, frame_samples: int, sample_rate: int) -> float:
        target = float(target)
        frame_ms = 1000.0 * frame_samples / float(sample_rate)
        tau = self.attack_ms if target < self.current else self.release_ms
        alpha = 1.0 - math.exp(-frame_ms / tau)
        self.current += alpha * (target - self.current)
        return self.current


class PeakLimiter:
    """Frame limiter used after local streams are mixed."""

    def __init__(
        self,
        threshold_dbfs: float = -3.0,
        min_gain_db: float = -24.0,
        attack_ms: float = 2.0,
        release_ms: float = 250.0,
    ):
        self.threshold_dbfs = float(threshold_dbfs)
        self.threshold_linear = 32768.0 * (10.0 ** (self.threshold_dbfs / 20.0))
        self.min_gain = 10.0 ** (float(min_gain_db) / 20.0)
        self.smoother = SmoothedGain(1.0, attack_ms, release_ms)

    def process_float(self, mixed: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict]:
        x = np.asarray(mixed, dtype=np.float32)
        if x.size == 0:
            return np.empty(0, dtype=np.int16), {
                "peak_dbfs": -120.0,
                "limiter_gain": 1.0,
                "limiter_active": False,
            }

        peak = float(np.max(np.abs(x)))
        if peak <= self.threshold_linear or peak <= 1e-9:
            target = 1.0
        else:
            target = max(self.min_gain, self.threshold_linear / peak)

        gain = self.smoother.update(target, x.size, sample_rate)
        y = np.clip(np.rint(x * gain), -32768, 32767).astype(np.int16)
        return y, {
            "peak_dbfs": round(peak_dbfs(np.clip(x, -32768, 32767).astype(np.int16)), 2),
            "limiter_gain": round(gain, 4),
            "limiter_active": gain < 0.999,
        }
