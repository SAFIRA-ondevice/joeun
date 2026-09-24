"""SAFIRA source routing policy.

A classifier score is NOT a separated waveform. The separated path in this
module only accepts real PCM streams produced by a source-separation stage.

Until a separator is available, classification_guided_ambient() may be used as
an explicit mixed-audio fallback. It cannot independently change simultaneous
drone and gunshot components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from self_speech import speech_routing_targets

from audio_dsp import (
    DRONE_GAIN,
    GUNSHOT_GAIN,
    SERVER_VOICE_GAIN,
    UNKNOWN_GAIN,
    USER_VOICE_GAIN,
    PeakLimiter,
    apply_linear_gain,
    mix_float_pcm16,
)


@dataclass
class SeparatedSources:
    gunshot: Optional[np.ndarray] = None
    drone: Optional[np.ndarray] = None
    unknown: Optional[np.ndarray] = None
    user_voice: Optional[np.ndarray] = None


class SafiraAudioRouter:
    def __init__(self, sample_rate: int = 16000, frame_samples: int = 320):
        self.sample_rate = int(sample_rate)
        self.frame_samples = int(frame_samples)
        self.limiter = PeakLimiter()

    def server_uplink(self, user_voice: np.ndarray) -> np.ndarray:
        """User voice x1.0. This is the stream sent to the server."""
        return apply_linear_gain(user_voice, USER_VOICE_GAIN)

    def headset_from_separated(
        self,
        sources: SeparatedSources,
        server_voice: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict]:
        """Mix only genuine separated PCM plus server downlink voice."""
        streams = []
        if sources.gunshot is not None:
            streams.append((sources.gunshot, GUNSHOT_GAIN))
        if sources.drone is not None:
            streams.append((sources.drone, DRONE_GAIN))
        if sources.unknown is not None:
            streams.append((sources.unknown, UNKNOWN_GAIN))
        if server_voice is not None:
            streams.append((server_voice, SERVER_VOICE_GAIN))

        mixed = mix_float_pcm16(streams, self.frame_samples)
        out, meta = self.limiter.process_float(mixed, self.sample_rate)
        meta.update({
            "routing_mode": "separated",
            "gunshot_gain": GUNSHOT_GAIN,
            "drone_gain": DRONE_GAIN,
            "unknown_gain": UNKNOWN_GAIN,
            "server_voice_gain": SERVER_VOICE_GAIN,
        })
        return out, meta

    def classification_guided_ambient(
        self,
        ambient_mix: np.ndarray,
        detected,
        server_voice: Optional[np.ndarray] = None,
        *,
        speech_state: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Fallback for the current classifier-only prototype.

        Because ambient_mix is not source-separated, only ONE scalar gain can
        be applied to it. Gunshot has protective priority over drone when both
        are detected. No target -> unknown policy -> mute.
        """
        # Speech attribution gates semantic eligibility, never cancels PCM.
        # No external-speech playback gain is specified in the current policy.
        routing_targets = speech_routing_targets(detected, speech_state or {})
        active = set(routing_targets)
        if "gunshot" in active:
            gain, reason = GUNSHOT_GAIN, "gunshot"
        elif "drone" in active:
            gain, reason = DRONE_GAIN, "drone"
        else:
            gain, reason = UNKNOWN_GAIN, "unknown"

        streams = [(ambient_mix, gain)]
        if server_voice is not None:
            streams.append((server_voice, SERVER_VOICE_GAIN))

        mixed = mix_float_pcm16(streams, self.frame_samples)
        out, meta = self.limiter.process_float(mixed, self.sample_rate)
        meta.update({
            "routing_mode": "classification_guided_mixed_fallback",
            "ambient_gain": gain,
            "ambient_reason": reason,
            "routing_detected": routing_targets,
            "self_speech_pcm_removed": False,
            "warning": "ambient PCM is not source-separated",
        })
        return out, meta
