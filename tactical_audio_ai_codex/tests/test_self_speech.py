import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "raspberry_pi"))
from self_speech import SelfSpeechConfig, SelfSpeechDetector, speech_routing_targets
from audio_routing import SafiraAudioRouter
from spk0 import pack_spk0, unpack_spk0


class SelfSpeechTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(61)
        self.voice = self.rng.normal(0, 4000, 16000)
        self.self_pcm = np.column_stack((np.roll(self.voice, 8) * .25,
                                        -np.roll(self.voice, 13) * .2, self.voice)).astype(np.int16)

    def update(self, pcm, detector=None, **kwargs):
        return (detector or SelfSpeechDetector()).update(
            pcm, ambient_speech_score=kwargs.pop("ambient_speech_score", .9),
            voice_speech_score=kwargs.pop("voice_speech_score", .9), **kwargs)

    def test_delayed_inverted_self_leakage(self):
        state = self.update(self.self_pcm)
        self.assertTrue(state["self_speech_active"])
        self.assertFalse(state["external_speech_candidate"])
        self.assertEqual(state["self_speech_metrics"]["lag_samples_lr"], [8, 13])
        self.assertFalse(state["self_speech_pcm_removed"])
        json.dumps(state, allow_nan=False)

    def test_independent_external_speech_with_quiet_or_busy_voice(self):
        for voice in (np.zeros(16000), self.voice):
            pcm = np.column_stack((self.rng.normal(0, 2000, (16000, 2)), voice))
            state = self.update(pcm)
            self.assertFalse(state["self_speech_active"])
            self.assertTrue(state["external_speech_candidate"])

    def test_silence_and_dc_cannot_create_speech(self):
        for level in (0, 2000):
            state = self.update(np.full((16000, 3), level))
            self.assertFalse(state["self_speech_active"])
            self.assertFalse(state["external_speech_candidate"])

    def test_correlated_non_speech_not_self(self):
        state = self.update(self.self_pcm, voice_speech_score=.1, ambient_speech_score=.1)
        self.assertFalse(state["self_speech_active"])
        self.assertFalse(state["external_speech_candidate"])

    def test_far_correlated_sound_is_uncertain(self):
        pcm = np.column_stack((self.voice, self.voice, self.voice * .25))
        state = self.update(pcm)
        self.assertFalse(state["self_speech_active"])
        self.assertFalse(state["external_speech_candidate"])
        self.assertEqual(state["speech_attribution"], "uncertain")

    def test_one_sided_leakage_and_no_lr_average_cancellation(self):
        for right in (self.rng.normal(0, 2000, 16000), -self.self_pcm[:, 0]):
            pcm = np.column_stack((self.self_pcm[:, 0], right, self.voice))
            self.assertTrue(self.update(pcm)["self_speech_active"])

    def test_smoothing_attack_and_hangover(self):
        detector = SelfSpeechDetector()
        state = self.update(self.self_pcm, detector, elapsed_seconds=.02)
        self.assertFalse(state["self_speech_active"])
        self.assertTrue(state["self_speech_raw_candidate"])
        self.assertFalse(state["external_speech_candidate"])
        state = self.update(self.self_pcm, detector, elapsed_seconds=.5)
        self.assertTrue(state["self_speech_active"])
        external = np.column_stack((self.rng.normal(0, 2000, (16000, 2)), np.zeros(16000)))
        self.assertTrue(self.update(external, detector)["self_speech_active"])
        state = self.update(external, detector)
        self.assertFalse(state["self_speech_active"])
        self.assertTrue(state["external_speech_candidate"])
        detector.reset()
        self.assertEqual(detector.smoothed_evidence, 0)
        self.assertEqual(detector.hold_seconds, 0)

    def test_annotation_preserves_ai_scores_and_gates_routing(self):
        result = {"detected": ["speech", "drone"], "targets": {"speech": 90.0}}
        SelfSpeechDetector().annotate(result, self.self_pcm, ["speech", "drone"],
                                     [.9, .8], [.95, .1], {"speech": .5}, .25)
        self.assertEqual(result["detected"], ["speech", "drone"])
        self.assertEqual(result["targets"]["speech"], 90)
        self.assertEqual(result["routing_detected"], ["drone"])
        self.assertEqual(result["voice_speech_score"], .95)
        self.assertFalse(result["source_separation_available"])

    def test_external_eligibility_uses_ambient_not_voice_score(self):
        pcm = np.column_stack((self.rng.normal(0, 2000, (16000, 2)), np.zeros(16000)))
        result = {"detected": []}
        SelfSpeechDetector().annotate(result, pcm, ["speech"], [.9], [.1], {"speech": .5}, .25)
        self.assertEqual(result["routing_detected"], ["external_speech"])
        self.assertEqual(speech_routing_targets(["speech", "external_speech"],
                         {"self_speech_active": True, "external_speech_candidate": True}), [])

    def test_invalid_inputs_and_configuration(self):
        for pcm in (np.zeros((320, 2)), np.zeros((319, 3)), np.full((320, 3), np.nan)):
            with self.assertRaises(ValueError):
                self.update(pcm)
        for kwargs in ({"correlation_threshold": 2}, {"max_lag_samples": -1},
                       {"smoothing_seconds": 0}, {"near_voice_ratio_db": float("nan")}):
            with self.assertRaises(ValueError):
                SelfSpeechConfig(**kwargs)
        with self.assertRaises(ValueError):
            self.update(self.self_pcm, voice_speech_score=float("nan"))

    def test_gain_and_server_path_unchanged_by_self_gate(self):
        ambient = np.full(320, 1000, np.int16)
        server = np.full(320, 2000, np.int16)
        state = self.update(self.self_pcm)
        for labels, gain in ((["speech"], 0), (["speech", "drone"], 1.5),
                             (["speech", "drone", "gunshot"], .15)):
            router = SafiraAudioRouter()
            output, meta = router.classification_guided_ambient(ambient, labels, server, speech_state=state)
            np.testing.assert_array_equal(output, np.full(320, 2000 + 1000 * gain, np.int16))
            self.assertNotIn("speech", meta["routing_detected"])
            packet = pack_spk0(7, output)
            self.assertEqual(len(packet), 648)
            np.testing.assert_array_equal(unpack_spk0(packet)[1], output)
            np.testing.assert_array_equal(router.server_uplink(ambient), ambient)


if __name__ == "__main__":
    unittest.main()
