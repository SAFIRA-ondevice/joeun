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
