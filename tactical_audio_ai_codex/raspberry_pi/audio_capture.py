"""Continuous single-reader M3C0 capture for latest-window inference.

Only the capture thread reads serial. Slow inference/logging never owns the
read loop. A bounded rolling window prioritizes freshness, not lossless storage.
Host timestamps do NOT measure microphone-to-ear latency.
"""
from collections import deque
from dataclasses import dataclass
import math
import threading
import time

import numpy as np

from m3c0 import sync_packet


@dataclass
class AudioWindow:
    pcm: np.ndarray
    elapsed_seconds: float
    generation: int
    received_at: float
    reset: bool


class ContinuousCapture:
    SAMPLE_RATE = 16000
    FRAME_SAMPLES = 320
    WINDOW_FRAMES = 50

    def __init__(self, port, hop_seconds=0.25):
        if not math.isfinite(hop_seconds) or hop_seconds <= 0:
            raise ValueError("hop_seconds must be positive and finite")
        self.port = port
        self.hop_frames = max(1, math.ceil(hop_seconds * 50))
        self._frames = deque(maxlen=self.WINDOW_FRAMES)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._expected = None
        self._total = 0
        self._last_taken = 0
        self._generation = 0
        self._last_generation = -1
        self._missing = 0
        self._discontinuities = 0
        self._backward = 0
        self._overwritten = 0
        self._skipped_hops = 0
        self._high_water = 0
        self._last_jump = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._receive, name="m3c0-reader", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            cancel = getattr(self.port, "cancel_read", None)
            if cancel is not None:
                try:
                    cancel()
                except (OSError, NotImplementedError):
                    pass
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                raise RuntimeError("M3C0 reader did not stop; close the serial port")

    def _receive(self):
        try:
            while not self._stop.is_set():
                seq, count, payload = sync_packet(self.port, channels=3)
                if count != self.FRAME_SAMPLES or len(payload) != 1920:
                    raise ValueError(f"expected 320 samples / 1920 bytes, got {count} / {len(payload)}")
                received_at = time.monotonic()
                with self._condition:
                    if self._stop.is_set():
                        return
                    if self._expected is not None and seq != self._expected:
                        delta = (seq - self._expected) & 0xFFFF
                        self._generation += 1
                        self._discontinuities += 1
                        # Forward distance is a gap estimate; backward/reset is
                        # ambiguous without a firmware timestamp/session ID.
                        if delta < 32768:
                            self._missing += delta
                        else:
                            self._backward += 1
                        self._last_jump = {"expected": self._expected, "received": seq}
                        self._frames.clear()
                    self._expected = (seq + 1) & 0xFFFF
                    self._total += 1
                    if len(self._frames) == self.WINDOW_FRAMES and self._frames[0][0] > self._last_taken:
                        self._overwritten += 1
                    self._frames.append((self._total, received_at, payload))
                    self._high_water = max(self._high_water, len(self._frames))
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                if not self._stop.is_set():
                    self._error = exc
                self._condition.notify_all()

    def next_window(self, timeout=2.0):
        """Take the latest contiguous second, or None while warming/rebuilding.

        Never replays queued inference jobs. After a gap, consumers reset their
        semantic history. Overload skips old analysis windows explicitly.
        """
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._error is not None or self._stop.is_set() or (
                    len(self._frames) == self.WINDOW_FRAMES
                    and self._total - self._last_taken >= self.hop_frames), timeout)
            if self._error is not None:
                raise RuntimeError(f"M3C0 capture failed: {self._error}") from self._error
            if not ready or self._stop.is_set():
                return None
            frames = list(self._frames)
            reset = self._last_generation != self._generation
            new_frames = self.WINDOW_FRAMES if reset else self._total - self._last_taken
            if not reset:
                self._skipped_hops += max(0, new_frames // self.hop_frames - 1)
            self._last_taken = self._total
            self._last_generation = self._generation
            generation = self._generation
        pcm = np.frombuffer(b"".join(frame[2] for frame in frames), dtype="<i2").reshape(-1, 3)
        return AudioWindow(pcm, new_frames * .02, generation, frames[-1][1], reset)

    def stats(self):
        with self._condition:
            now = time.monotonic()
            return {
                "packets_received": self._total,
                "missing_frames_estimate": self._missing,
                "sequence_discontinuities": self._discontinuities,
                "backward_or_restart_events": self._backward,
                "last_sequence_jump": self._last_jump,
                "generation": self._generation,
                "window_frames": len(self._frames),
                "window_capacity_frames": self.WINDOW_FRAMES,
                "window_high_water_frames": self._high_water,
                "overwritten_unconsumed_frames": self._overwritten,
                "analysis_hops_skipped": self._skipped_hops,
                "oldest_frame_host_age_ms": round((now - self._frames[0][1]) * 1000, 1) if self._frames else None,
                "newest_frame_host_age_ms": round((now - self._frames[-1][1]) * 1000, 1) if self._frames else None,
            }
