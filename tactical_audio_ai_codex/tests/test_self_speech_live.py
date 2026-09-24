"""Exercise both live entry points with synthetic transport and mocked CNN.

These tests do not validate a checkpoint, hardware, or real-time performance.
"""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

PI = Path(__file__).parents[1] / "raspberry_pi"
sys.path.insert(0, str(PI))
from spk0 import unpack_spk0


class LiveAttributionTests(unittest.TestCase):
    def run_live(self, name, jump=False):
        dependencies = {name: ModuleType(name) for name in ("torch", "torchaudio", "serial", "audio_model")}
        dependencies["torch"].load = MagicMock(return_value={
            "task": "multilabel", "classes": ["speech", "drone", "gunshot"], "state_dict": {},
        })
        dependencies["audio_model"].AudioCNN = MagicMock()
        serial = MagicMock()
        serial.__enter__.return_value = serial
        dependencies["serial"].Serial = MagicMock(return_value=serial)
        # Load the actual companion module as well, under mocked heavy imports.
        with patch.dict(sys.modules, dependencies):
            spec = importlib.util.spec_from_file_location("live_m3c0_multilabel", PI / "live_m3c0_multilabel.py")
            companion = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(companion)
            with patch.dict(sys.modules, {"live_m3c0_multilabel": companion}):
                spec = importlib.util.spec_from_file_location(name, PI / (name + ".py"))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

        rng = np.random.default_rng(3)
        voice = rng.normal(0, 4000, 320).astype(np.int16)
        pcm = np.column_stack((voice // 4, voice // 5, voice)).astype("<i2")
        packets = [(i + (1 if jump and i >= 50 else 0), 320, pcm.tobytes()) for i in range(75)]
        args = [name, "--model", "mock.pt", "--no-start-command", "--json"]
        full_duplex = name.endswith("full_duplex")
        if full_duplex:
            args += ["--headset-serial", "--server-host", "127.0.0.1", "--uplink-port", "9001", "--downlink-port", "9002"]
        output = io.StringIO()
        sender, receiver = MagicMock(), MagicMock()
        receiver.recv.side_effect = BlockingIOError
        with patch.object(sys, "argv", args), patch.object(module, "sync_packet", side_effect=packets), \
             patch.object(module, "infer", return_value=np.array([.95, .1, .1])), redirect_stdout(output):
            if full_duplex:
                with patch.object(module, "Spk0UdpSender", return_value=sender), \
                     patch.object(module, "Spk0UdpReceiver", return_value=receiver):
                    with self.assertRaises(StopIteration):
                        module.main()
            else:
                with self.assertRaises(StopIteration):
                    module.main()
        results = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(results), 1 if jump else 2)
        for result in results:
            self.assertTrue(result["self_speech_active"])
            self.assertFalse(result["external_speech_candidate"])
            self.assertEqual(result["routing_detected"], [])
            self.assertEqual(result["detected"], ["speech"])
            self.assertFalse(result["source_separation_available"])
            self.assertFalse(result["self_speech_pcm_removed"])
        if full_duplex:
            self.assertEqual(sender.send.call_count, 75)
            for call in sender.send.call_args_list:
                np.testing.assert_array_equal(call.args[0], voice)
            self.assertEqual(serial.write.call_count, 75)
            for call in serial.write.call_args_list:
                self.assertEqual(len(call.args[0]), 648)
                self.assertFalse(unpack_spk0(call.args[0])[1].any())
            sender.close.assert_called_once()
            receiver.close.assert_called_once()

    def test_multilabel_entrypoint(self):
        self.run_live("live_m3c0_multilabel")

    def test_full_duplex_entrypoint(self):
        self.run_live("live_m3c0_full_duplex")

    def test_sequence_gap_resets_both_inference_windows(self):
        for name in ("live_m3c0_multilabel", "live_m3c0_full_duplex"):
            with self.subTest(name=name):
                self.run_live(name, jump=True)


if __name__ == "__main__":
    unittest.main()
