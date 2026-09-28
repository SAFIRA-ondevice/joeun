"""Deterministic serial simulation: the real parser and receive thread run.

Events pause the inference consumer while the fake device supplies more PCM.
No hardware, CPU-load claim, or click-removal assertion is made here.
"""
import struct
import sys
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "raspberry_pi"))
from audio_capture import ContinuousCapture


class FakeSerial:
    def __init__(self):
        self.buffer = bytearray()
        self.condition = threading.Condition()
        self.waiting = threading.Event()
        self.cancelled = False
        self.failure = None
        self.read_threads = set()

    def feed(self, sequences):
        with self.condition:
            self.waiting.clear()
            for seq in sequences:
                payload = np.full((320, 3), seq % 30000, dtype="<i2").tobytes()
                self.buffer.extend(struct.pack("<4sHH", b"M3C0", seq & 65535, 320) + payload)
            self.condition.notify_all()

    def read(self, size):
        self.read_threads.add(threading.get_ident())
        with self.condition:
            if not self.buffer:
                self.waiting.set()
                self.condition.wait_for(lambda: self.buffer or self.cancelled or self.failure, timeout=2)
            if self.failure:
                raise self.failure
            if self.cancelled:
                return b""
            # Deliberately fragmented reads exercise the existing M3C0 parser.
            size = min(size, 127)
            data = bytes(self.buffer[:size])
            del self.buffer[:size]
            return data

    def cancel_read(self):
        with self.condition:
            self.cancelled = True
            self.condition.notify_all()

    def fail(self):
        with self.condition:
            self.failure = OSError("device disconnected")
            self.condition.notify_all()


class CaptureTests(unittest.TestCase):
    def test_slow_inference_keeps_reading_and_bounds_memory(self):
        port = FakeSerial()
        with ContinuousCapture(port) as capture:
            port.feed(range(50))
            first = capture.next_window()
            self.assertEqual(first.pcm.shape, (16000, 3))
            self.assertTrue(first.reset)
            # Consumer intentionally does not ask for another inference window
            # until 120 additional packets have been consumed by the reader.
            port.feed(range(50, 170))
            self.assertTrue(port.waiting.wait(2), "capture stalled with inference idle")
            stats = capture.stats()
            self.assertEqual(stats["packets_received"], 170)
            self.assertEqual(stats["window_frames"], 50)
            self.assertEqual(stats["window_high_water_frames"], 50)
            self.assertEqual(stats["overwritten_unconsumed_frames"], 70)
            latest = capture.next_window()
            self.assertEqual(latest.pcm[0, 0], 120)
            self.assertEqual(latest.pcm[-1, 0], 169)
            self.assertAlmostEqual(latest.elapsed_seconds, 2.4)
            self.assertFalse(latest.reset)
            self.assertGreater(capture.stats()["analysis_hops_skipped"], 0)
            self.assertEqual(len(port.read_threads), 1)
            self.assertNotIn(threading.get_ident(), port.read_threads)
        self.assertFalse(capture._thread.is_alive())

    def test_gap_rebuilds_contiguous_window_and_counts_large_jump(self):
        port = FakeSerial()
        with ContinuousCapture(port) as capture:
            port.feed(range(50))
            capture.next_window()
            port.feed(range(2050, 2099))
            self.assertTrue(port.waiting.wait(2))
            self.assertIsNone(capture.next_window(timeout=.01))
            port.feed([2099])
            window = capture.next_window()
            self.assertTrue(window.reset)
            self.assertEqual(window.pcm[0, 0], 2050)
            self.assertEqual(capture.stats()["missing_frames_estimate"], 2000)
            self.assertEqual(capture.stats()["sequence_discontinuities"], 1)

    def test_wrap_is_continuous_but_backward_jump_is_reported(self):
        port = FakeSerial()
        with ContinuousCapture(port) as capture:
            port.feed(range(65510, 65560))
            capture.next_window()
            self.assertEqual(capture.stats()["sequence_discontinuities"], 0)
            port.feed(range(10, 60))
            self.assertTrue(port.waiting.wait(2))
            self.assertTrue(capture.next_window().reset)
            stats = capture.stats()
            self.assertEqual(stats["backward_or_restart_events"], 1)
            self.assertEqual(stats["missing_frames_estimate"], 0)

    def test_reader_error_reaches_consumer(self):
        port = FakeSerial()
        with ContinuousCapture(port) as capture:
            port.fail()
            with self.assertRaisesRegex(RuntimeError, "device disconnected"):
                capture.next_window()
        self.assertFalse(capture._thread.is_alive())

    def test_close_cancels_blocked_read(self):
        port = FakeSerial()
        with ContinuousCapture(port) as capture:
            self.assertTrue(port.waiting.wait(2))
        self.assertTrue(port.cancelled)
        self.assertFalse(capture._thread.is_alive())

    def test_invalid_frame_is_reported(self):
        port = FakeSerial()
        with port.condition:
            port.buffer.extend(struct.pack("<4sHH", b"M3C0", 0, 1) + b"\0" * 6)
        with ContinuousCapture(port) as capture:
            with self.assertRaisesRegex(RuntimeError, "expected 320 samples"):
                capture.next_window()


if __name__ == "__main__":
    unittest.main()
