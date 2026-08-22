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

# 2) 학습
python training/train_audio_cnn.py --data splits --output models/audio_cnn_4class_best.pt

# 3) 실시간 추론
python raspberry_pi/live_m3c0_ai.py --port /dev/ttyUSB1 --model models/audio_cnn_4class_best.pt
```

상세 현황과 다음 작업은 `CODEX_CONTINUE.md`를 먼저 읽으세요.

## 구조

- `training/`: 정규화, 균형 분할, MAD 라벨 추출, background 구축, CNN 학습
- `raspberry_pi/`: M3C0 WAV 캡처와 실시간 추론
- `models/`: 모델 기대 경로와 메타데이터
- `docs/`: 프로토콜 및 데이터셋 메모

데이터셋은 라이선스와 용량 문제로 포함하지 않았습니다.
