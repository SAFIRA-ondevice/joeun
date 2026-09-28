#!/usr/bin/env python3
"""Experimental delayed live separation. Save beside source_separation.py.
No network uplink, speaker verification, or hearing-protection guarantee.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import struct
import threading
import time
import wave
import sys

import numpy as np


class ActivityGate:
    """3 dB level hysteresis + 320 ms release, measured in input frames."""
    def __init__(self):
        self.key = None
        self.active = False
        self.until = -1

    def update(self, key, peak, threshold):
        if key is None:
            return peak > threshold
        if self.key is None or key[0] != self.key[0] or not 0 < key[1]-self.key[1] <= 50:
            self.active = False
            self.until = -1
        self.key = key
        cutoff = threshold-3 if self.active else threshold
        if peak > cutoff:
            self.active = True
            self.until = key[1]+16
        elif key[1] >= self.until:
            self.active = False
        return self.active


class MutePolicy:
    def __init__(self, combined_gain=1.):
        self.combined_gain = combined_gain
        self.generation = None
        self.self_until = -1
        self.combined_until = -1
        self.hits = 0
        self.last_key = None

    def update(self, key, self_fraction, detected):
        if self.generation != key[0]:
            self.generation = key[0]
            self.self_until = self.combined_until = -1
            self.hits = 0
        if self.last_key is None or key != (self.last_key[0],self.last_key[1]+4):
            self.hits = 0
        self.last_key = key
        self.hits = self.hits+1 if self_fraction >= .60 else 0
        if self_fraction >= .80 or self.hits >= 2:
            self.self_until = key[1]+6  # 120 ms release hold instead of 400 ms
        if {'gunshot','drone'}.issubset(set(detected)):
            self.combined_until = key[1]+20
        return key[1] <= self.self_until, self.combined_gain if key[1] <= self.combined_until else 1.


def mix_frame(stem, gains, ramp, muted):
    if muted or stem is None:
        return np.zeros(320,dtype=np.float32)
    return (stem*gains[:,None]).sum(0)*ramp


class NoiseFloor:
    """Level gate on detection labels, also used by the combined attenuation policy."""
    def __init__(self, margin=6.):
        self.margin = margin
        self.levels = []
        self.threshold = None
        self.floor = None
        self.last_key = None
        self.low_frames = 0
        self.adjustments = []
        self.activity = ActivityGate()

    def observe(self, ambient):
        self.levels.append(dbfs(ambient))

    def finish(self):
        if len(self.levels) < 100:
            raise ValueError('잡음 보정 실패: 3초간 유효 프레임이 부족합니다')
        # Discard opening 0.5 s; high clicks/speech must not define quiet floor.
        self.floor = float(np.percentile(self.levels[25:],25))
        self.threshold = max(-90., self.floor+self.margin)
        return {'noise_floor_dbfs':self.floor,'activity_threshold_dbfs':self.threshold,
                'margin_db':self.margin,'frames':len(self.levels), 'affects_combined_detection_gain':True,
                'estimator':'25th_percentile_after_first_500ms',
                'calibration_spread_db':float(np.percentile(self.levels[25:],90)-self.floor)}

    def apply(self, ambient, text, classification, key=None):
        if self.threshold is None:
            raise RuntimeError('Noise calibration incomplete')
        levels = [dbfs(frame) for frame in ambient.reshape(-1,320)]
        if key is not None:
            delta = key[1]-self.last_key[1] if self.last_key is not None and key[0]==self.last_key[0] else 0
            if not 0 < delta <= 50:
                self.low_frames = 0
            elif np.percentile(levels,75) < self.floor-8:
                self.low_frames += delta
            else:
                self.low_frames = 0
            self.last_key = key
            if self.low_frames >= 100:  # 2 s of newly received audio, not overlapping-window count
                old = self.floor
                self.floor = max(-96.,float(np.percentile(levels,25)))
                self.threshold = max(-90.,self.floor+self.margin)
                self.adjustments.append({'frame_index':key[1],'old_floor_dbfs':old,'new_floor_dbfs':self.floor})
                self.low_frames = 0
        peak = max(levels)
        quiet = not self.activity.update(key,peak,self.threshold)
        result = dict(classification)
        result.update(input_peak_frame_dbfs=peak, input_rms_dbfs=dbfs(ambient),
                      noise_floor_dbfs=self.floor, activity_threshold_dbfs=self.threshold,
                      below_activity_threshold=quiet,
                      below_entry_threshold=peak <= self.threshold,
                      activity_release_threshold_dbfs=self.threshold-3,
                      level_gate_hold_ms=320,
                      model_detected_before_level_gate=list(classification['detected']))
        if quiet:
            result['detected'] = []
            text = '[무음/배경 수준] 감지 보류 | 원래 분류점수 '+ ' '.join(
                f"{name} {classification['scores'].get(c,0)*100:.0f}%"
                for c,name in [('gunshot','총'),('drone','드론'),('speech','말')])
        held = ' 감지유지' if not quiet and peak <= self.threshold else ''
        return text+f' | 입력최대 {peak:.1f} 진입 {self.threshold:.1f} 해제 {self.threshold-3:.1f}dBFS{held}', result


def classification_status(classes, probabilities, thresholds):
    names = {'speech': '주변 말소리', 'drone': '드론', 'gunshot': '총소리'}
    scores = {str(c).lower(): float(v) for c, v in zip(classes, probabilities)}
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in scores.values()):
        raise ValueError('Invalid classification scores')
    detected = [str(c).lower() for c, v in zip(classes, probabilities)
                if str(c).lower() in names and float(v) >= thresholds[c]]
    label = ' · '.join(names[c]+' 감지' for c in detected) or '감지 기준 미만'
    detail = ' '.join(f'{names[c]} {scores[c]*100:.0f}%' for c in ('gunshot','drone','speech') if c in scores)
    return f'[{label}] 분류점수 {detail}', {'scores': scores, 'detected': detected,
            'source': 'ambient_left', 'scores_are_calibrated_accuracy': False}


class AmbientClassifier:
    def __init__(self, path):
        import torch
        import torchaudio
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'training'))
        from audio_model import AudioCNN
        ck = torch.load(path, map_location='cpu', weights_only=True)
        if ck.get('task') != 'multilabel' or int(ck.get('sample_rate',16000)) != 16000:
            raise ValueError('classifier-model requires the 16 kHz multilabel checkpoint')
        self.classes = ck['classes']
        if not {'speech','drone','gunshot'}.issubset({str(c).lower() for c in self.classes}):
            raise ValueError('Classifier requires speech, drone and gunshot classes')
        self.thresholds = {c: float(ck.get('thresholds',{}).get(c,.5)) for c in self.classes}
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in self.thresholds.values()):
            raise ValueError('Invalid classifier thresholds')
        self.model = AudioCNN(len(self.classes), **ck.get('model_config',{}))
        self.model.load_state_dict(ck['state_dict'])
        self.model.eval()
        self.mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=512,
            win_length=400, hop_length=160, n_mels=int(ck.get('n_mels',64)), f_min=20, f_max=7600)
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.history = deque(maxlen=3)
        self.key = None

    def infer(self, ambient, key, hop_frames=25):
        import torch
        if self.key is None or key != (self.key[0], self.key[1]+hop_frames):
            self.history.clear()
        self.key = key
        with torch.no_grad():
            z = self.to_db(self.mel(torch.from_numpy(ambient.astype(np.float32))))
            z = (z-z.mean())/(z.std()+1e-6)
            probability = torch.sigmoid(self.model(z[None,None]))[0].numpy()
        self.history.append(probability)
        text, result = classification_status(self.classes, np.mean(self.history,axis=0), self.thresholds)
        result['raw_scores'] = {str(c).lower(): float(v) for c,v in zip(self.classes,probability)}
        return text, result

def suppress_self(speech, voice_speech, ambient, voice, strength=.8, profile=None, output_region=None):
    arrays = [np.asarray(v, dtype=np.float32) for v in (speech, voice_speech, ambient, voice)]
    n = arrays[0].size
    if n < 320 or n % 320 or any(v.shape != (n,) or not np.isfinite(v).all() for v in arrays):
        raise ValueError('Expected equal finite arrays in 320-sample frames')
    if not 0 <= strength <= 1:
        raise ValueError('strength must be in [0,1]')
    speech, ref, ambient, voice = arrays
    # Search +/-3 ms. This is an acoustic-reference heuristic, not voiceprint ID.
    best, delay = 0., 0
    lags = range(-48, 49) if profile is None else range(max(-48, profile['lag_samples']-8), min(48, profile['lag_samples']+8)+1)
    for lag in lags:
        x = ref[max(0, -lag):min(n, n-lag)]
        y = speech[max(0, lag):min(n, n+lag)]
        corr = float(np.dot(x, y) / np.sqrt(max(float(np.dot(x,x)*np.dot(y,y)), 1e-16)))
        if abs(corr) > abs(best):
            best, delay = corr, lag
    shifted = np.zeros_like(ref)
    lo, hi = max(0, delay), min(n, n+delay)
    shifted[lo:hi] = ref[lo-delay:hi-delay]
    mask, predicted = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    evidence = np.zeros(n,dtype=np.float32)
    near_ratios = []
    speech_fractions = []
    for start in range(0, n, 320):
        sl = slice(start, start+320)
        x, y = shifted[sl], speech[sl]
        vp = float(np.mean(voice[sl]**2))
        ap = float(np.mean(ambient[sl]**2))
        near = 10*np.log10((vp+1e-12)/(ap+1e-12))
        near_ratios.append(float(near))
        xp, yp = float(np.dot(x,x)), float(np.dot(y,y))
        corr = float(np.dot(x,y)/np.sqrt(max(xp*yp, 1e-16)))
        # Reject quiet reference, far sources, and non-speech-dominant reference.
        speech_fraction = float(np.mean(ref[sl]**2))/(vp+1e-12)
        speech_fractions.append(float(speech_fraction))
        near_threshold = 6 if profile is None else profile['near_threshold_db']
        if vp > 1e-8 and near >= near_threshold and speech_fraction >= .35 and abs(corr) >= .65:
            coefficient = float(np.dot(x,y)/(xp+1e-8))
            predicted[sl] = coefficient*x
            mask[sl] = strength
            evidence[sl] = 1
    # Smooth on/off and blockwise estimates across 10 ms boundaries.
    kernel = np.ones(161, dtype=np.float32)/161
    mask = np.convolve(mask, kernel, mode='same')
    # Projection subtraction is conservative; no whole-speech muting.
    removed = predicted*mask
    external = speech-removed
    return external, removed, {'self_reference_active': bool(np.max(mask) > .1),
        'output_self_fraction':float(np.mean(evidence[slice(*output_region)] if output_region else evidence)),
        'active_fraction': float(np.mean(mask > .1)), 'reference_correlation': best,
        'reference_lag_samples': delay, 'near_ratio_db_median': float(np.median(near_ratios)), 'speech_fraction_median': float(np.median(speech_fractions)),
        'method': 'gated_reference_projection', 'speaker_identity_verified': False}


def validate_profile(profile):
    if not isinstance(profile, dict) or profile.get('kind') != 'wearer_acoustic_calibration':
        raise ValueError('Invalid wearer profile kind')
    near, lag = profile.get('near_threshold_db'), profile.get('lag_samples')
    if isinstance(near, bool) or not isinstance(near, (int, float)) or not np.isfinite(near) or not 1 <= near <= 18:
        raise ValueError('Invalid near_threshold_db')
    if isinstance(lag, bool) or not isinstance(lag, int) or not -48 <= lag <= 48:
        raise ValueError('Invalid lag_samples')
    return profile


def make_profile(rows):
    corr_ok = sum(abs(r['reference_correlation']) >= .65 for r in rows)
    near_ok = sum(r['near_ratio_db_median'] >= 1 for r in rows)
    voice_ok = sum(r['voice_dbfs'] > -45 for r in rows)
    speech_ok = sum(r['voice_speech_fraction'] >= .35 for r in rows)

    print(
        f"[등록 DEBUG] 전체={len(rows)} "
        f"corr통과={corr_ok} "
        f"near통과={near_ok} "
        f"voice통과={voice_ok} "
        f"speechFrac통과={speech_ok}",
        flush=True
    )

    usable = [r for r in rows if abs(r['reference_correlation']) >= .65
              and r['near_ratio_db_median'] >= 1 and r['voice_dbfs'] > -45
              and r['voice_speech_fraction'] >= .35]

    print(f"[등록 DEBUG] 최종 usable={len(usable)} / {len(rows)}", flush=True)
    # Five-second enrollment uses overlapping one-second analysis windows.
    if len(usable) < 5:
        raise ValueError('본인 등록 실패: 유효한 근접 음성이 부족합니다. 입 마이크 위치를 확인하고 주변 재생을 끈 뒤 다시 5초 말해주세요.')
    ratios = [r['near_ratio_db_median'] for r in usable]
    lags = [r['reference_lag_samples'] for r in usable]
    if np.percentile(lags, 90)-np.percentile(lags, 10) > 20:
        raise ValueError('본인 등록 실패: 마이크 간 지연이 불안정합니다. 헤드셋 위치를 고정하고 다시 등록해주세요.')
    return {'kind': 'wearer_acoustic_calibration', 'speaker_identity_verified': False,
            'near_threshold_db': float(np.clip(np.percentile(ratios, 10)-2, 1, 18)),
            'lag_samples': int(round(float(np.median(lags)))), 'accepted_hops': len(usable),
            'near_ratio_median_db': float(np.median(ratios)), 'enrollment_seconds': 5}


def dbfs(x):
    return float(10*np.log10(max(float(np.mean(np.asarray(x)**2)), 1e-12)))



class Overlap:
    def __init__(self, samples=16000):
        self.samples = samples
        self.half = samples//2
        self.hop_frames = self.half//320
        self.tail = None
        self.key = None
        self.fade = np.linspace(0, 1, self.half, dtype=np.float32)

    def process(self, stems, key):
        if stems.shape != (4, self.samples) or not np.isfinite(stems).all():
            raise ValueError("Invalid separator output")
        out = stems[:, :self.half].copy()
        if self.tail is not None and key == (self.key[0], self.key[1] + self.hop_frames):
            out = self.tail * (1 - self.fade) + out * self.fade
        self.tail, self.key = stems[:, self.half:].copy(), key
        return out


class GunDuck:
    """Attenuates ALL output on gunshot classification; not source separation."""
    def __init__(self, factor):
        self.factor = factor
        self.until = -1
        self.generation = None

    def update(self, score, threshold, key):
        if self.generation != key[0]:
            self.until = -1
            self.generation = key[0]
        if score >= threshold:
            self.until = key[1]+25  # 0.5 s of input timeline
        return self.factor if key[1] <= self.until else 1.


def recent_chunk(stems, hop_samples, lookahead_samples):
    if stems.shape != (4,16000) or not np.isfinite(stems).all():
        raise ValueError('Expected 1-second separated context')
    end = stems.shape[1]-lookahead_samples
    return stems[:, end-hop_samples:end].copy()


def packet(sequence, x):
    if np.shape(x) != (320,) or not np.isfinite(x).all():
        raise ValueError("Invalid output frame")
    pcm = (np.clip(x, -1, 1) * 32767).round().astype('<i2')
    return struct.pack('<4sHH', b'SPK0', sequence & 65535, 320) + pcm.tobytes()


def frames(buffer):
    result = []
    while True:
        pos = buffer.find(b'M3C0')
        if pos < 0:
            if len(buffer) > 3:
                del buffer[:-3]
            break
        if pos:
            del buffer[:pos]
        if len(buffer) < 8:
            break
        seq, count = struct.unpack_from('<HH', buffer, 4)
        if count != 320:
            del buffer[:4]
            continue
        if len(buffer) < 1928:
            break
        x = np.frombuffer(bytes(buffer[8:1928]), dtype='<i2').reshape(320, 3).astype(np.float32) / 32768
        del buffer[:1928]
        result.append((seq, x))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--classifier-model', type=Path, required=True)
    p.add_argument('--port', default='/dev/ttyUSB0')
    p.add_argument('--baud', type=int, default=1000000)
    p.add_argument('--attach-speaker', action='store_true', help='Only if ESP32 already streams in speaker mode')
    p.add_argument('--seconds', type=int, default=120)
    p.add_argument('--speech-gain', type=float, default=1.5)
    p.add_argument('--drone-gain', type=float, default=1.5)
    p.add_argument('--gunshot-gain', type=float, default=.15)
    p.add_argument('--unknown-gain', type=float, default=1.5)
    p.add_argument('--master', type=float, default=.01)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--self-strength', type=float, default=1.)
    p.add_argument('--lookahead-ms', type=int, choices=(40,80,120), default=80)
    p.add_argument('--noise-margin-db', type=float, default=6.)
    p.add_argument('--combined-gain', type=float, default=1.)
    a = p.parse_args()
    if not np.isfinite(a.combined_gain) or not 0 <= a.combined_gain <= 1:
        p.error('combined-gain must be 0..1')
    if not np.isfinite(a.noise_margin_db) or not 0 <= a.noise_margin_db <= 20:
        p.error('noise-margin-db must be 0..20')
    samples = 16000
    half = 1280  # emit 80 ms of recent audio, rather than the oldest half-window
    hop_frames = 4
    lookahead = a.lookahead_ms*16
    slice_end = samples-lookahead
    slice_start = slice_end-half
    if not 0 <= a.self_strength <= 1:
        p.error('self-strength must be 0..1')
    if not 1 <= a.seconds <= 300 or not 0 <= a.master <= 1 or a.threads < 1:
        p.error('seconds 1..300, master 0..1, threads >=1 required')
    gains = np.array([a.speech_gain, a.drone_gain, a.gunshot_gain, a.unknown_gain], dtype=np.float32)
    if not np.isfinite(gains).all() or (gains < 0).any() or (gains > 10).any():
        p.error('gains must be finite and in 0..10')
    import serial
    import torch
    from runtime_separator import Separator
    torch.set_num_threads(a.threads)
    checkpoint = torch.load(a.model, map_location='cpu', weights_only=True)
    if checkpoint.get('task') != 'source_separation_v1':
        p.error('A separator checkpoint is required, not a classifier')
    model = Separator().eval()
    model.load_state_dict(checkpoint['state_dict'])
    classifier = AmbientClassifier(a.classifier_model)
    with torch.no_grad():
        model(torch.zeros(1, 16000))
        model(torch.zeros(1, samples))
    lock = threading.Lock()
    event, stop = threading.Event(), threading.Event()
    state = {'job': None, 'generation': 0, 'skipped': 0, 'missing': 0,
             'underrun': 0, 'dropped': 0, 'processed': 0, 'error': None, 'urgent_mute':False, 'muted_frames':0}
    playback = deque()
    beep_playback = deque()
    recordings, estimates, mixed_recordings = [], [], []
    comparisons, metrics = [], []
    enrollment_audio, enrollment_rows = [], []

    PROFILE_PATH = Path(__file__).resolve().parent / 'wearer_profile.json'
    REQUEST_PATH = Path(__file__).resolve().parent / '.reenroll_request'

    try:
        profile = json.loads(PROFILE_PATH.read_text(encoding='utf-8')) if PROFILE_PATH.exists() else None
        if profile is not None:
            validate_profile(profile)
            print(f"[프로파일 로드] 기준 {profile['near_threshold_db']:.1f}dB, 지연 {profile['lag_samples']}샘플", flush=True)
    except Exception as exc:
        print(f"[프로파일 로드 실패] {exc}", flush=True)
        profile = None

    state['reenroll_phase'] = 'idle'       # idle/start_beep/settle/enrolling/finalize
    state['reenroll_started'] = 0.0
    state['settle_until'] = 0.0

    def queue_beep(count=1, hz=880.0):
        # Raw PCM is queued here but written only by the existing main SPK0 loop.
        for beep_index in range(count):
            t = np.arange(int(16000 * .12), dtype=np.float32) / 16000.0
            tone = (.08 * np.sin(2*np.pi*hz*t)).astype(np.float32)
            for start in range(0, tone.size, 320):
                frame = np.zeros(320, dtype=np.float32)
                part = tone[start:start+320]
                frame[:part.size] = part
                beep_playback.append(frame)
            if beep_index != count-1:
                for _ in range(4):
                    beep_playback.append(np.zeros(320, dtype=np.float32))

    noise = NoiseFloor(a.noise_margin_db)
    noise_audio = []
    noise_report = None
    folder = Path('recordings') / ('live_stable_' + time.strftime('%Y%m%d_%H%M%S'))
    folder.mkdir(parents=True, exist_ok=False)

    def infer():
        nonlocal profile
        classification_cache = None
        classified_key = None
        policy = MutePolicy(a.combined_gain)
        try:
            while not stop.is_set():
                event.wait(.2)
                with lock:
                    job, state['job'] = state['job'], None
                    event.clear()
                if job is None:
                    continue
                key, x, created, enrolling, elapsed = job
                began = time.monotonic()
                classification_input = x[:, 0].copy()
                if not enrolling:
                    x = x[-samples:]
                ambient = x[:, 0].copy()
                voice = x[:, 2].copy()
                with torch.no_grad():
                    result = model(torch.from_numpy(np.stack([ambient, voice])))[0].numpy()
                with lock:
                    if key[0] != state['generation']:
                        continue
                stems = result[0].copy()
                finalize_now = (not enrolling and state['reenroll_phase'] == 'finalize')
                if finalize_now:
                    # Build a candidate first. A failed enrollment never destroys the active profile.
                    try:
                        candidate = make_profile(enrollment_rows)
                        validate_profile(candidate)
                        tmp = PROFILE_PATH.with_suffix('.json.tmp')
                        tmp.write_text(json.dumps(candidate, indent=2), encoding='utf-8')
                        tmp.replace(PROFILE_PATH)
                        profile = candidate
                        try:
                            (folder / 'wearer_profile.json').write_text(json.dumps(profile, indent=2), encoding='utf-8')
                        except OSError as exc:
                            print(f'[프로파일 녹음폴더 복사 실패] {exc}', flush=True)
                        print(f"[재등록 완료] 기준 {profile['near_threshold_db']:.1f}dB, 지연 {profile['lag_samples']}샘플", flush=True)
                        queue_beep(2, 1000.0)
                    except Exception as exc:
                        print(f"[재등록 실패] {exc} | 기존 프로파일 유지", flush=True)
                        queue_beep(3, 440.0)
                    finally:
                        state['reenroll_phase'] = 'idle'

                external, removed, metric = suppress_self(stems[0], result[1, 0], ambient, voice, a.self_strength, profile,
                    None if enrolling else (slice_start,slice_end))
                # With no saved profile, keep the AI running but do not suppress/mute speech as "self".
                if profile is None and not enrolling:
                    external = stems[0].copy()
                    removed = np.zeros_like(stems[0])
                    metric['output_self_fraction'] = 0.0
                    metric['self_reference_active'] = False
                metric['voice_dbfs'] = dbfs(voice)
                metric['voice_speech_fraction'] = float(np.mean(result[1,0]**2)/(np.mean(voice**2)+1e-12))
                if enrolling:
                    with lock:
                        if key[0] != state['generation']:
                            continue
                        enrollment_rows.append(metric)
                    print(f"[재등록 {min(5, int(elapsed))}/5초] 본인만 계속 말해주세요 | 입 {metric['voice_dbfs']:.0f}dBFS", flush=True)
                    continue
                # Classification is display-only and runs every 320 ms. Audio
                # separation emits a new recent block every 80 ms independently.
                if classified_key is None or key[0] != classified_key[0] or key[1]-classified_key[1] >= 16:
                    classification_cache = classifier.infer(classification_input, key, 16)
                    classified_key = key
                classification_text, classification = classification_cache
                classification_text, classification = noise.apply(classification_input,classification_text,classification,key)
                metric['classification'] = classification
                metric['classification_age_ms'] = (key[1]-classified_key[1])*20
                hard_mute, duck = policy.update(key, metric['output_self_fraction'],classification['detected'])
                metric['whole_output_self_mute'] = hard_mute
                metric['whole_output_duck'] = duck
                original_speech = stems[0].copy()
                stems[0] = external
                out = recent_chunk(stems,half,lookahead)
                with lock:
                    if key[0] != state['generation'] or time.monotonic() - created > .16:
                        state['dropped'] += hop_frames
                        continue
                    estimates.append(out.copy())
                    state['urgent_mute'] = hard_mute
                    comparisons.append(np.stack([v[slice_start:slice_end] for v in (ambient,voice,original_speech,external,removed)]))
                    metrics.append(dict(metric, generation=key[0], frame_index=key[1]))
                    for frame in out.reshape(4, hop_frames, 320).transpose(1, 0, 2):
                        if len(playback) >= hop_frames+1:
                            playback.popleft()
                            state['dropped'] += 1
                        playback.append((frame.copy(), duck, hard_mute))
                    state['processed'] += 1
                    if key == classified_key:
                        print(f"[SELF DEBUG] corr={metric['reference_correlation']:.3f} lag={metric['reference_lag_samples']} near={metric['near_ratio_db_median']:.1f}dB speechFrac={metric['speech_fraction_median']:.2f} self={metric['output_self_fraction']*100:.0f}%", flush=True)
                        print(f"{classification_text} | 최근 본인근거 {metric['output_self_fraction']*100:.0f}% | {'본인 후보: 전체 출력 0' if hard_mute else f'혼합 추가 gain x{duck:g}'} | 분리 gain 외부말 x{gains[0]:g} 드론 x{gains[1]:g} 총 x{gains[2]:g} 미분류 x{gains[3]:g} | "
                          f"분리신호 말 {dbfs(stems[0]):.0f} 드론 {dbfs(stems[1]):.0f} 총 {dbfs(stems[2]):.0f}dBFS | "
                          f"처리 {(time.monotonic()-began)*1000:.0f}ms | 누락 {state['missing']} "
                          f"출력대기 {state['underrun']} 지연폐기 {state['dropped']}", flush=True)
        except Exception as exc:
            with lock:
                state['error'] = repr(exc)
            stop.set()

    port = serial.Serial(port=None, baudrate=a.baud, timeout=.003, write_timeout=.2)
    port.dtr = False
    port.rts = False
    port.port = a.port
    worker = None
    try:
        port.open()
        if not a.attach_speaker:
            print('ESP32 준비 대기 2초...', flush=True)
            time.sleep(2)
            port.write(b'START_INMP_SPEAKER\n')
            line = bytearray()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                b = port.read(1)
                line.extend(b)
                if b == b'\n':
                    if bytes(line).strip(b'\x00\r\n ').startswith(b'STARTED'):
                        break
                    line.clear()
                if len(line) > 4096:
                    line.clear()
            else:
                raise RuntimeError('STARTED ACK 없음. 이전에 speaker 모드로 실행했다면 --attach-speaker로 재연결하세요. 입력 전용 모드는 해당하지 않습니다.')
        worker = threading.Thread(target=infer, daemon=True)
        worker.start()
        print('먼저 3초간 주변 잡음 기준을 측정합니다. 이후 AI가 바로 실행됩니다.', flush=True)
        if profile is None:
            print('[프로파일 없음] AI는 실행됩니다. 필요할 때 별도 등록 요청을 보내세요.', flush=True)
        else:
            print('[프로파일 준비] 저장된 사용자 프로파일을 사용합니다.', flush=True)
        print(f'과거 1초 문맥 유지, 80ms마다 최근 구간 출력. 입력 구간 지연 {80+a.lookahead_ms}ms + 처리/송신/장치 지연. 실제 전체 지연 미측정.', flush=True)
        buffer, window = bytearray(), deque(maxlen=50)
        expected, index, tx = None, 0, 0
        began = last_rx = last_tx = time.monotonic()
        previous = np.zeros(320, dtype=np.float32)
        limiter = 1.0
        duck_gain = 1.0
        while time.monotonic() - began < a.seconds+3 and not stop.is_set():
            buffer.extend(port.read(max(1, min(port.in_waiting, 8192))))
            for seq, x in frames(buffer):
                last_rx = time.monotonic()
                if expected is not None and seq != expected:
                    with lock:
                        state['missing'] += (seq - expected) & 65535
                        state['generation'] += 1
                        state['job'] = None
                        playback.clear()
                        state['urgent_mute'] = False
                    window.clear()
                expected = (seq + 1) & 65535
                if last_rx-began < 3:
                    noise.observe(x[:, 0])
                    noise_audio.append(x.copy())
                    continue
                if noise.threshold is None:
                    noise_report = noise.finish()
                    print(f'[잡음 측정 완료] 바닥 {noise.floor:.0f} 기준 {noise.threshold:.0f}dBFS. AI 정상 동작 시작.', flush=True)
                    window.clear()

                # A separate helper (later: GPIO watcher) only creates this request file.
                # This process remains the sole owner of the serial/audio stream.
                if REQUEST_PATH.exists() and state['reenroll_phase'] == 'idle':
                    try:
                        REQUEST_PATH.unlink()
                    except FileNotFoundError:
                        pass
                    with lock:
                        state['generation'] += 1
                        enrollment_rows.clear()
                    enrollment_audio.clear()
                    window.clear()
                    with lock:
                        state['job'] = None
                        playback.clear()
                        state['urgent_mute'] = False
                    state['reenroll_phase'] = 'start_beep'
                    queue_beep(1, 880.0)
                    print('[재등록 요청] 시작 비프 후 5초간 본인 음성을 등록합니다.', flush=True)

                # Start enrollment only after the beep has left the output queue,
                # then wait 250 ms so the beep is not used as reference data.
                if state['reenroll_phase'] == 'start_beep' and not beep_playback:
                    state['reenroll_phase'] = 'settle'
                    state['settle_until'] = time.monotonic() + .25
                    window.clear()

                if state['reenroll_phase'] == 'settle' and time.monotonic() >= state['settle_until']:
                    state['reenroll_phase'] = 'enrolling'
                    state['reenroll_started'] = time.monotonic()
                    window.clear()
                    print('[재등록 시작] 지금부터 본인만 5초간 말해주세요.', flush=True)

                enrolling_now = state['reenroll_phase'] == 'enrolling'
                if enrolling_now:
                    enrollment_audio.append(x.copy())
                    if time.monotonic() - state['reenroll_started'] >= 5.0:
                        state['reenroll_phase'] = 'finalize'
                        enrolling_now = False
                        window.clear()
                        print('[재등록 수집 완료] 새 프로파일 검증 중...', flush=True)

                # During start beep/settle/enrollment/finalize, normal playback is muted.
                if state['reenroll_phase'] != 'idle':
                    with lock:
                        playback.clear()
                        state['urgent_mute'] = False

                window.append(x.copy())
                index += 1
                job_hop = 25 if enrolling_now else hop_frames
                if len(window) == 50 and index % job_hop == 0:
                    with lock:
                        if state['job'] is not None:
                            state['skipped'] += 1
                        elapsed = time.monotonic() - state['reenroll_started'] if enrolling_now else 0.0
                        state['job'] = ((state['generation'], index), np.concatenate(window), last_rx, enrolling_now, elapsed)
                    event.set()
            now = time.monotonic()
            if now - last_rx > 3:
                raise TimeoutError('3초 동안 유효한 M3C0 프레임이 없습니다')
            # Monotonic pacing: never burst queued speaker packets after a stall.
            if now - last_tx >= .02:
                last_tx = last_tx + .02 if now - last_tx < .04 else now
                with lock:
                    item = playback.popleft() if playback else None
                    active = state['processed'] > 0
                    urgent_mute = state['urgent_mute']
                    if item is None and active:
                        state['underrun'] += 1
                stem, target_duck, queued_mute = item if item is not None else (None, duck_gain, False)
                hard_mute = urgent_mute or queued_mute
                next_duck = min(target_duck, duck_gain+.08)
                duck_ramp = np.linspace(duck_gain,next_duck,320)
                duck_gain = next_duck
                mixed = mix_frame(stem,gains,duck_ramp,hard_mute)
                if hard_mute:
                    state['muted_frames'] += 1
                peak = float(np.abs(mixed).max())
                required = min(1., .9 / max(peak, 1e-9))
                limiter = min(required, limiter + .01)
                y = mixed * limiter * a.master
                mixed_recordings.append((mixed*limiter).copy())

                # Beeps use the same SPK0 writer and sequence counter; no second serial writer.
                if beep_playback:
                    y = beep_playback.popleft() * a.master
                elif state['reenroll_phase'] != 'idle':
                    y = np.zeros(320, dtype=np.float32)
                # Short boundary ramp also softens underrun transitions.
                if not hard_mute:
                    y[:32] = previous[-1] * np.linspace(1, 0, 32) + y[:32] * np.linspace(0, 1, 32)
                port.write(packet(tx, y))
                tx += 1
                previous = y
                recordings.append(y.copy())
        if state['error']:
            raise RuntimeError(state['error'])
    except KeyboardInterrupt:
        print('종료, 녹음 저장 중…', flush=True)
    finally:
        stop.set()
        event.set()
        if worker is not None:
            worker.join()
        if port.is_open:
            try:
                port.write(packet(0, np.zeros(320)))
            except Exception:
                pass
            finally:
                port.close()
        def save(name, x):
            with wave.open(str(folder / name), 'wb') as f:
                f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                f.writeframes((np.clip(x, -1, 1) * 32767).round().astype('<i2').tobytes())
        if recordings:
            save('headset_sent.wav', np.concatenate(recordings))
            save('processed_before_master.wav', np.concatenate(mixed_recordings))
        if enrollment_audio:
            audio = np.concatenate(enrollment_audio)
            with wave.open(str(folder / 'enrollment_3ch.wav'), 'wb') as f:
                f.setparams((3, 2, 16000, 0, 'NONE', 'not compressed'))
                f.writeframes((np.clip(audio,-1,1)*32767).round().astype('<i2').tobytes())
        if noise_audio:
            audio = np.concatenate(noise_audio)
            with wave.open(str(folder / 'noise_calibration_3ch.wav'),'wb') as f:
                f.setparams((3,2,16000,0,'NONE','not compressed'))
                f.writeframes((np.clip(audio,-1,1)*32767).round().astype('<i2').tobytes())
        (folder / 'enrollment_metrics.json').write_text(json.dumps(enrollment_rows, indent=2), encoding='utf-8')
        if comparisons:
            comparison = np.concatenate(comparisons, axis=1)
            for i, name in enumerate(('ambient_before', 'voice_reference', 'speech_before', 'external_speech_candidate', 'self_removed_estimate')):
                save(name+'.wav', comparison[i])
        (folder / 'self_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        if estimates:
            all_stems = np.concatenate(estimates, axis=1)
            for i, name in enumerate(('speech', 'drone', 'gunshot', 'background')):
                save(name + '_estimate.wav', all_stems[i])
                save(name + '_after_gain.wav', all_stems[i]*gains[i])
            mixed = (all_stems * gains[:, None]).sum(0)
            save('diagnostic_before_mute.wav', mixed * min(1., .9 / max(float(np.abs(mixed).max()), 1e-9)))
        state.pop('job', None)
        (folder / 'report.json').write_text(json.dumps({'model': str(a.model), 'gains': gains.tolist(),
             'master': a.master, 'stats': state, 'self_vs_external_speech_separation': False,
             'self_reference_suppression_enabled': True, 'self_strength': a.self_strength,
             'wearer_profile': profile, 'enrollment_type': 'acoustic_calibration_not_voiceprint',
             'classifier_model': str(a.classifier_model), 'classification_controls_gain': a.combined_gain != 1.0,
             'context_ms': 1000, 'emit_ms':80, 'lookahead_ms':a.lookahead_ms,
             'noise_calibration':noise_report,
             'noise_final_floor_dbfs':noise.floor, 'noise_floor_adjustments':noise.adjustments,
             'self_mute_policy':'recent_output_80pct_or_two_60pct_hops_hold120ms',
             'whole_output_gun_duck': 1.,
             'simultaneous_gun_drone_gain':a.combined_gain, 'self_candidate_mutes_all_sources':True,
             'quality_verified_for_live_use': False}, indent=2), encoding='utf-8')
        print(f'결과 폴더: {folder}', flush=True)


if __name__ == '__main__':
    main()
