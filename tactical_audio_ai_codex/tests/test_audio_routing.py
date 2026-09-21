import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "raspberry_pi"))

from audio_dsp import DRONE_GAIN, GUNSHOT_GAIN
from audio_routing import SafiraAudioRouter, SeparatedSources
from spk0 import FRAME_SAMPLES, PACKET_BYTES, pack_spk0, unpack_spk0


class Spk0Tests(unittest.TestCase):
    def test_packet_size_and_roundtrip(self):
        x = np.arange(FRAME_SAMPLES, dtype=np.int16)
        packet = pack_spk0(65535, x)
        self.assertEqual(len(packet), PACKET_BYTES)
        seq, y = unpack_spk0(packet)
        self.assertEqual(seq, 65535)
        np.testing.assert_array_equal(x, y)


class RoutingTests(unittest.TestCase):
    def test_user_voice_uplink_is_unity_gain(self):
        router = SafiraAudioRouter()
        voice = np.full(FRAME_SAMPLES, 1234, dtype=np.int16)
        np.testing.assert_array_equal(router.server_uplink(voice), voice)

    def test_separated_gain_policy_without_limiter_activation(self):
        router = SafiraAudioRouter()
        gunshot = np.full(FRAME_SAMPLES, 1000, dtype=np.int16)
        drone = np.full(FRAME_SAMPLES, 1000, dtype=np.int16)
        out, meta = router.headset_from_separated(
            SeparatedSources(gunshot=gunshot, drone=drone)
        )
        expected = round(1000 * GUNSHOT_GAIN + 1000 * DRONE_GAIN)
        self.assertTrue(np.all(out == expected))
        self.assertEqual(meta["routing_mode"], "separated")

    def test_mixed_fallback_gunshot_has_priority(self):
        router = SafiraAudioRouter()
        ambient = np.full(FRAME_SAMPLES, 1000, dtype=np.int16)
        out, meta = router.classification_guided_ambient(
            ambient, ["drone", "gunshot"]
        )
        self.assertTrue(np.all(out == round(1000 * GUNSHOT_GAIN)))
        self.assertEqual(meta["ambient_reason"], "gunshot")

    def test_unknown_is_muted_without_server_voice(self):
        router = SafiraAudioRouter()
        ambient = np.full(FRAME_SAMPLES, 1000, dtype=np.int16)
        out, meta = router.classification_guided_ambient(ambient, [])
        self.assertTrue(np.all(out == 0))
        self.assertEqual(meta["ambient_reason"], "unknown")


if __name__ == "__main__":
    unittest.main()
