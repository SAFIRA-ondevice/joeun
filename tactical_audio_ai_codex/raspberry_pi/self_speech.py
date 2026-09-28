"""Enrollment-free speech attribution heuristics; never modifies/removes PCM.

Compare synchronized 20 ms chunks within the SAME window used by the CNN.
The output is a routing hint, not verified speaker identity or separation.
"""
from dataclasses import dataclass, asdict
import json
import math

import numpy as np


@dataclass(frozen=True)
class SelfSpeechConfig:
    sample_rate: int = 16000
    frame_samples: int = 320
    max_lag_samples: int = 32
    correlation_threshold: float = 0.65
    voice_min_dbfs: float = -40.0
    ambient_min_dbfs: float = -55.0
    near_voice_ratio_db: float = 6.0
    evidence_fraction: float = 0.6
    smoothing_seconds: float = 0.25
    hangover_seconds: float = 0.5

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("self-speech settings must be finite")
        if self.sample_rate != 16000 or self.frame_samples != 320:
            raise ValueError("self-speech requires 16 kHz / 320 sample frames")
        if not isinstance(self.max_lag_samples, int) or not 0 <= self.max_lag_samples < 160:
            raise ValueError("max_lag_samples must be an integer in [0, 159]")
        if not 0 < self.correlation_threshold <= 1 or not 0 < self.evidence_fraction <= 1:
            raise ValueError("correlation_threshold and evidence_fraction must be in (0, 1]")
        if self.smoothing_seconds <= 0 or self.hangover_seconds < 0:
            raise ValueError("invalid smoothing/hangover time")


def add_self_speech_arguments(parser):
    parser.add_argument("--self-speech-config", help="JSON overrides for SelfSpeechConfig (no enrollment)")


def detector_from_args(args):
    overrides = {}
    if args.self_speech_config:
        with open(args.self_speech_config, encoding="utf-8") as source:
            overrides = json.load(source)
    return SelfSpeechDetector(SelfSpeechConfig(**overrides))


def _dbfs(x):
    return 20 * math.log10(max(float(np.sqrt(np.mean(x * x))) / 32768, 1e-6))


def _correlation(reference, ambient, max_lag):
    """Max absolute normalized correlation; positive lag means ambient is later."""
    # Vectorized overlap normalization avoids a Python loop over every lag.
    n = len(reference)
    lags = np.arange(-max_lag, max_lag + 1)
    starts_x, starts_y = np.maximum(-lags, 0), np.maximum(lags, 0)
    counts = n - np.abs(lags)

    def moments(x, starts):
        sums = np.concatenate(([0.0], np.cumsum(x)))
        squares = np.concatenate(([0.0], np.cumsum(x * x)))
        total = sums[starts + counts] - sums[starts]
        energy = squares[starts + counts] - squares[starts] - total * total / counts
        return total, np.maximum(energy, 0)

    sum_x, energy_x = moments(reference, starts_x)
    sum_y, energy_y = moments(ambient, starts_y)
    products = np.correlate(ambient, reference, mode="full")[n - 1 + lags]
    denominator = np.sqrt(energy_x * energy_y)
    values = np.divide(np.abs(products - sum_x * sum_y / counts), denominator,
                       out=np.zeros_like(denominator), where=denominator > 1e-9)
    index = int(np.argmax(values))
    return min(1.0, float(values[index])), int(lags[index]) if values[index] > 0 else 0


def speech_routing_targets(detected, state):
    """Raw CNN `speech` is voice-mic speech, never external speech by itself."""
    targets = [c for c in detected if c.lower() not in {"speech", "external_speech"}]
    if state.get("external_speech_candidate", False) and not state.get("self_speech_active", False):
        targets.append("external_speech")
    return targets


class SelfSpeechDetector:
    def __init__(self, config=None):
        self.config = config or SelfSpeechConfig()
        self.reset()

    def reset(self):
        self.smoothed_evidence = 0.0
        self.hold_seconds = 0.0

    def update(self, pcm, *, ambient_speech_score, voice_speech_score,
               speech_threshold=0.5, elapsed_seconds=0.25):
        """pcm is the synchronized Nx3 PCM16-scale CNN window, not separated PCM.

        Call at inference cadence with unsmoothed scores for this exact window.
        elapsed_seconds is NEW captured audio since the previous update, not
        window length: overlapping windows must not inflate hangover time.
        """
        cfg = self.config
        pcm = np.asarray(pcm, dtype=np.float64)
        if pcm.ndim != 2 or pcm.shape[1] != 3 or len(pcm) == 0 or len(pcm) % cfg.frame_samples:
            raise ValueError("expected nonempty Nx3 audio in complete 320-sample frames")
        if not np.isfinite(pcm).all() or np.max(np.abs(pcm)) > 32768:
            raise ValueError("audio must be finite PCM16-scale samples")
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be positive and finite")
        for score in (ambient_speech_score, voice_speech_score, speech_threshold):
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("speech scores/threshold must be in [0, 1]")

        features = []
        for chunk in pcm.reshape(-1, cfg.frame_samples, 3):
            centered = chunk - chunk.mean(axis=0)
            levels = [_dbfs(centered[:, i]) for i in range(3)]
            matches = [_correlation(centered[:, 2], centered[:, i], cfg.max_lag_samples)
                       for i in range(2)]
            # Each side must satisfy correlation AND proximity together.
            leakage = any(levels[i] >= cfg.ambient_min_dbfs
                          and levels[2] - levels[i] >= cfg.near_voice_ratio_db
                          and matches[i][0] >= cfg.correlation_threshold for i in range(2))
            strong_voice = levels[2] >= cfg.voice_min_dbfs
            features.append((levels, matches, strong_voice and leakage))

        levels = np.asarray([f[0] for f in features])
        audible = np.max(levels[:, :2], axis=1) >= cfg.ambient_min_dbfs
        votes = np.asarray([f[2] for f in features])
        fraction = float(votes[audible].mean()) if audible.any() else 0.0
        voice_speech = voice_speech_score >= speech_threshold
        ambient_speech = ambient_speech_score >= speech_threshold and bool(audible.any())
        evidence = fraction if voice_speech and ambient_speech else 0.0
        alpha = 1 - math.exp(-elapsed_seconds / cfg.smoothing_seconds)
        self.smoothed_evidence += alpha * (evidence - self.smoothed_evidence)
        raw_self = evidence >= cfg.evidence_fraction
        if raw_self and self.smoothed_evidence >= cfg.evidence_fraction:
            self.hold_seconds = cfg.hangover_seconds
            active = True
        else:
            self.hold_seconds = max(0.0, self.hold_seconds - elapsed_seconds)
            active = self.hold_seconds > 0
        # Also suppress external attribution while self evidence is attacking.
        # Correlated but non-near speech is ambiguous, not confidently external.
        correlations = np.asarray([[m[0] for m in f[1]] for f in features])
        correlated_fraction = float((correlations[audible].max(axis=1) >= cfg.correlation_threshold).mean()) if audible.any() else 0.0
        external = ambient_speech and not active and not raw_self and correlated_fraction < cfg.evidence_fraction
        return {
            "self_speech_active": bool(active),
            "external_speech_candidate": bool(external),
            "speech_attribution": "self_candidate" if active else "external_candidate" if external else "uncertain" if ambient_speech else "no_ambient_speech",
            "speech_attribution_method": "voice_reference_heuristic",
            "self_speech_evidence": round(self.smoothed_evidence, 4),
            "self_speech_raw_candidate": bool(raw_self),
            "ambient_speech_score": float(ambient_speech_score),
            "voice_speech_score": float(voice_speech_score),
            "self_speech_metrics": {
                "correlation_lr": correlations.mean(axis=0).tolist(),
                "lag_samples_lr": np.median([[m[1] for m in f[1]] for f in features], axis=0).tolist(),
                "rms_dbfs_lrv": levels.mean(axis=0).tolist(),
                "near_voice_ratio_db_lr": (levels[:, 2, None] - levels[:, :2]).mean(axis=0).tolist(),
                "leakage_fraction": fraction,
                "correlated_fraction": correlated_fraction,
                "window_samples": len(pcm),
            },
            "self_speech_pcm_removed": False,
        }

    def annotate(self, result, pcm, classes, ambient_prob, voice_prob, thresholds, elapsed_seconds):
        index = next((i for i, c in enumerate(classes) if c.lower() == "speech"), None)
        if index is None:
            raise ValueError("self-speech attribution requires the speech model class")
        state = self.update(pcm, ambient_speech_score=float(ambient_prob[index]),
                            voice_speech_score=float(voice_prob[index]),
                            speech_threshold=thresholds[classes[index]], elapsed_seconds=elapsed_seconds)
        result.update(state)
        result["routing_detected"] = speech_routing_targets(result["detected"], state)
        result["source_separation_available"] = False
        return state
