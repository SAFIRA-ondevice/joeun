# Tactical Audio AI — Codex continuation package

Raspberry Pi 5와 ESP32-WROOM-32D(INMP441 3개)를 이용하는 실시간 전술 음향 분류 프로토타입입니다.

> 중요: 최초 3-class 모델의 `gunshot` 데이터에는 MAD의 7개 클래스 전체가 섞였습니다. 이 모델의 정확도는 유효한 평가값이 아니며 배포하면 안 됩니다. MAD에서 실제 gunshot만 추출하고 `background`를 추가한 4-class 모델을 다시 학습하세요.

## 빠른 시작 (Raspberry Pi)

```bash
cd /home/pi/tactical_audio_ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) 원천 WAV를 data_raw/{speech,drone,gunshot,background}에 준비
python training/prepare_dataset.py --source data_raw --output dataset --classes speech drone gunshot background
python training/split_dataset.py --input dataset --output splits --classes speech drone gunshot background --balance-min

# 2) 복합음 multi-label 학습
python training/train_multilabel_cnn.py --data splits --output models/audio_cnn_multilabel_best.pt

# 기존 단일 선택 baseline이 필요할 때만 사용
python training/train_audio_cnn.py --data splits --output models/audio_cnn_4class_best.pt

# 3) 각 대상의 독립 확률 출력
python raspberry_pi/live_m3c0_multilabel.py \
  --port /dev/ttyUSB1 \
  --model models/audio_cnn_multilabel_best.pt

# 4) 대상이 없을 때 AudioSet/YAMNet 521종으로 보조 추정
pip install -r requirements-audioset.txt
python raspberry_pi/live_m3c0_multilabel.py \
  --port /dev/ttyUSB1 \
  --model models/audio_cnn_multilabel_best.pt \
  --yamnet
```

상세 현황과 다음 작업은 `CODEX_CONTINUE.md`를 먼저 읽으세요.

## 구조

- `training/`: 정규화, 균형 분할, MAD 라벨 추출, background 구축, CNN 학습
- `raspberry_pi/`: M3C0 WAV 캡처와 실시간 추론
- `models/`: 모델 기대 경로와 메타데이터
- `docs/`: 프로토콜 및 데이터셋 메모

데이터셋은 라이선스와 용량 문제로 포함하지 않았습니다.

출력 예시:

```text
[AI] speech= 81.2% | drone= 74.5% | gunshot=  3.1% | => SPEECH + DRONE
[AI] speech=  4.2% | drone=  8.1% | gunshot=  1.9% | => 추정: Siren
[AI] speech=  2.0% | drone=  3.1% | gunshot=  1.2% | => UNKNOWN
```

퍼센트는 서로 독립이므로 합이 100%일 필요가 없습니다. 운영 전 클래스별 threshold를 별도 검증 세트로 보정해야 합니다.


## SAFIRA DSP / full-duplex routing

최신 프로토타입은 3번째 `voice` 마이크를 speech 판정과 사용자 음성 uplink에 사용합니다.
환경음은 ambient L/R 평균을 사용합니다.

증감/라우팅 정책:

- separated gunshot ×0.15 → local headset
- separated drone ×1.50 → local headset
- user voice ×1.00 → server
- unknown ×0.00 → local headset
- server voice ×1.00 → local headset

통신 오디오 규격은 `16 kHz / S16LE / mono / 20 ms / 320 samples / 640 bytes`이고,
UDP 전송 시 `SPK0` 8-byte header를 붙여 총 648 bytes입니다.

현재 CNN은 source separation 모델이 아니라 multi-label classifier입니다. 따라서 실제 분리 PCM이
없는 상태에서는 `classification_guided_mixed_fallback`을 사용하며, 이를 separated audio라고
표시하지 않습니다. 실제 separator가 추가되면 `SafiraAudioRouter.headset_from_separated()`
경로로 교체합니다.

관련 파일:

- `raspberry_pi/audio_dsp.py`
- `raspberry_pi/audio_routing.py`
- `raspberry_pi/audio_network.py`
- `raspberry_pi/spk0.py`
- `raspberry_pi/live_m3c0_full_duplex.py`
- `docs/SAFIRA_AUDIO_ROUTING.md`

### Enrollment 없는 자기 음성 판단

두 live 스크립트는 `raspberry_pi/self_speech.py`로 voice reference와 ambient L/R의
동기화된 신호를 비교합니다. 시작 시 개인 목소리를 녹음·학습하지 않습니다.
근접 voice 에너지, 좌우 각각의 정규화 상관관계와 시간차, 같은 구간의 CNN speech
점수를 사용하며 smoothing/hangover를 적용합니다. 초기 1초 분석 구간 이후
`--hop` 주기(기본 약 250 ms, 20 ms 프레임 단위)로 상태를 갱신합니다.

- `targets.speech` / `detected`: 기존 voice 마이크 기반 AI 결과 그대로 유지
- `self_speech_active`: 자기 음성 유입 후보 및 hangover 상태
- `external_speech_candidate`: 자기 음성 gating을 통과한 외부 음성 후보
- `routing_detected`: 일반 `speech`를 제거하고 조건을 만족할 때만 `external_speech` 추가
- `self_speech_metrics`: 상관관계·지연·에너지 진단값, score와 구분됨
- `self_speech_pcm_removed=false`: PCM에서 자기 목소리를 제거한 것이 아님

외부 speech의 헤드셋 gain은 현재 정책에 정의되어 있지 않아 자동 재생을 추가하지
않습니다. 혼합 PCM에 drone/gunshot gain이 적용되면 그 안의 자기 음성도 함께
변합니다. 이를 해결하려면 별도 source separation/self-voice cancellation이 필요합니다.
동시 발화, 반사음, 마이크 gain 차이 등에서는 오판할 수 있습니다.

`--self-speech-config settings.json`으로 검출 threshold 등을 보정할 수 있습니다.
설정 키와 실기 검증 항목은 [라우팅 문서](docs/SAFIRA_AUDIO_ROUTING.md)에 있습니다.
합성 신호/모의 I/O 테스트는 다음과 같이 실행합니다. 실기 end-to-end 검증은 미완료입니다.

```bash
python3 -m unittest discover -s tests -v
```

### AI 실행 주기에 맞춰 클릭음이 날 때: 연속 수신 비교

`live_m3c0_multilabel.py`는 serial 수신을 전용 스레드로 분리했습니다.
AI 추론 및 로그 출력 중에도 수신을 계속하고, 최대 50프레임(1초)만 보관하여
최신 연속 구간을 추론합니다. 느린 추론 때문에 지나간 분석 구간은 다시 처리하지
않습니다. 무손실 녹음용 경로가 아닙니다.

이 실행 파일의 기본 baud는 최신 ESP32 인계 기준인 **1000000**이며,
`--torch-threads 1`이 기본입니다. CPU 추론 병렬도를 줄여 수신 여유를 확보하려는
설정이지 실시간 성능을 보장하는 설정은 아닙니다. 다른 진입점의 기본 baud는
이번 수정에 포함되지 않았으므로 사용 시 `--baud 1000000`을 명시하세요.

현재 프로그램을 종료하고 ESP32 EN으로 재시작한 뒤 약 2초 기다려 실행합니다.
다른 serial monitor는 닫아야 합니다.

```bash
python -u raspberry_pi/live_m3c0_multilabel.py --port /dev/ttyUSB0 --baud 1000000 --model /home/pi/tactical_audio_ai_codex/models/multilabel_10epoch.pt --hop 0.25 --torch-threads 1 --json
```

`audio_rx`에 수신 패킷 수, sequence 불연속, 누락 추정치, 버퍼 사용량,
미사용 프레임 교체 및 건너뛴 분석 횟수가 나옵니다. `analysis_window_age_ms`는
마지막 프레임을 호스트에서 받은 뒤 결과를 출력하기 직전까지의 시간이며 입→귀
지연이 아닙니다. 누락 시 연속 1초를 다시 모으고 기존 분류/자기 음성 상태를
초기화합니다. 추론 중 새 불연속이 발생하면 결과에
`input_discontinuity_during_inference=true`를 표시합니다.

이 수정은 **AI 수신 전용 스크립트**에 적용했습니다. `live_m3c0_full_duplex.py`의
동기식 처리 및 ESP32 speaker-mode 시작/ACK 문제는 별도 후속 작업입니다.
실기에서 클릭음 제거는 아직 검증하지 않았습니다. 같은 조건에서 무음/말하기와
hop 0.25/1.0을 비교하고, 소리와 `audio_rx` 로그를 함께 확인해야 합니다.
