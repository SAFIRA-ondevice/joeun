# 환경음 AI 구현 사양

## 1단계: 저장 WAV 분류

입력은 16 kHz mono PCM16 WAV다. 우선 `ambient_left`와 `ambient_right` 각각을 분류하고, 이후 두 채널 평균 mono도 비교한다.

초기 모델 후보는 YAMNet이다. YAMNet은 16 kHz mono float waveform을 입력으로 받고 AudioSet의 다양한 환경음 클래스를 출력한다.

우선 감시할 개념:

- Gunshot, gunfire
- Machine gun
- Fusillade
- Explosion
- Boom
- Siren
- Civil defense siren
- Fire alarm
- Smoke detector, smoke alarm
- Glass
- Shatter
- Screaming
- Crying, sobbing
- Vehicle horn, car horn, honking

프로그램 출력 예:

```text
[2026-08-23 16:20:10] top=Gunshot, gunfire score=0.87
ALERT type=gunshot label=총소리 confidence=0.87
```

JSONL 예:

```json
{"timestamp":"2026-08-23T16:20:10+09:00","event":"gunshot","label_ko":"총소리","confidence":0.87,"source":"ambient_mix"}
```

## 2단계: 실시간 M3C0 연결

- 매 M3C0 payload에서 3채널 PCM16을 분리한다.
- ambient left/right 평균을 mono float32 `[-1, 1]`로 변환한다.
- 최근 약 1초(16,000 samples)를 ring buffer에 유지한다.
- 약 0.5초마다 추론한다.
- 최근 N회 점수를 평균한다.
- threshold 이상이 consecutive 횟수만큼 지속될 때만 이벤트를 발생시킨다.
- 동일 이벤트 재발송을 막기 위한 cooldown을 둔다.

권장 CLI:

```bash
python3 realtime_sound_classifier.py \
  --port /dev/ttyUSB1 \
  --baud 921600 \
  --threshold 0.55 \
  --consecutive 2 \
  --cooldown 5 \
  --jsonl logs/sound_events.jsonl
```

`--baud` 기본값은 고정하지 말고 현재 ESP32 펌웨어와 맞출 수 있게 한다.

## 완료 기준

- 저장 WAV 한 개를 넣으면 상위 5개 클래스와 점수를 출력한다.
- 무음/일상 소음에서 위험 경보를 남발하지 않는다.
- 실시간 수신 중 패킷 누락과 resync 횟수를 함께 표시한다.
- Ctrl+C로 WAV와 JSONL을 손상 없이 닫는다.
- 모델 미설치, 포트 없음, baud 불일치에 이해 가능한 오류를 출력한다.
