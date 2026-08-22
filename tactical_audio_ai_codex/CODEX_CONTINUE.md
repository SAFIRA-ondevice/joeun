# CODEX_CONTINUE

## 현재 상태

- Raspberry Pi 작업 경로: `/home/pi/joeun`(기존 수신/추론 코드), `/home/pi/tactical_audio_ai`(데이터·학습·모델)
- 하드웨어: Raspberry Pi 5 8GB, ESP32-WROOM-32D, INMP441 × 3
- 채널 순서: `ambient_left`, `ambient_right`, `voice`
- 직렬 연결: USB-UART, 921600 baud
- 오디오: 16 kHz, signed PCM16 little-endian, 채널당 320 sample(20 ms)
- 패킷: magic `M3C0` + sequence `uint16 LE` + sample_count `uint16 LE` + 3채널 interleaved PCM16
- 시작 명령/응답: Pi가 `START_INMP_ONLY\n`, ESP32가 `STARTED\n`
- 추론 입력: ambient L/R 평균 mono, 최근 1초, 0.25초 hop, 64-bin log-mel
- 기존 Pi 5 추론 시간: 대개 30–40 ms, 일부 50–60 ms

## 발견된 치명적 데이터 문제

기존 baseline은 speech/drone/gunshot 각 1,332개로 구성했지만 `gunshot` 1,332개를 MAD 전체에서 무작위 선택했습니다. MAD에는 communication, gunshot, footsteps, shelling, vehicle, helicopter, fighter가 함께 있으므로 gunshot 라벨이 오염되었습니다. 사람 음성·무음도 gunshot으로 예측한 로그는 이 문제와 3-class softmax가 무조건 한 클래스를 고르는 문제로 설명됩니다.

## 모델 파일 상태

이 패키지를 만든 환경에서 실제 모델 바이너리를 찾지 못했습니다. 다음 경로를 Raspberry Pi에서 확인하세요.

```text
/home/pi/tactical_audio_ai/models/audio_cnn_best.pt          # 오염된 기존 3-class 후보; 평가/배포 금지
/home/pi/tactical_audio_ai/models/audio_cnn_4class_best.pt   # 새로 생성해야 할 권장 모델
```

체크포인트는 이 패키지의 학습 코드가 저장하는 `state_dict`, `classes`, `sample_rate`, `n_mels`, `model_config`를 포함해야 합니다.

## 정확한 다음 단계

1. Pi에서 위 기존 모델과 원본 코드를 찾아 이 패키지와 비교·보관한다.
2. MAD annotation 파일 구조를 확인하고 `extract_mad_gunshot.py`를 `--dry-run`으로 실행한다.
3. 실제 gunshot만 `data_raw/gunshot`으로 복사한다. MAD 전체 폴더를 복사하지 않는다.
4. DroneAudioDataset의 non-drone 22,076개 후보와 무음/실내/바람/차량/모터를 `build_background.py`로 구성한다.
5. `prepare_dataset.py`, `split_dataset.py --balance-min`으로 leakage 없는 균형 분할을 만든다.
6. 4-class 모델(speech, drone, gunshot, background)을 학습한다.
7. test split confusion matrix와 실제 마이크(말/무음/선풍기/드론/총성)로 검증한다.
8. 새 체크포인트를 `audio_cnn_4class_best.pt`로 연결한다.
9. 이후 helicopter/fighter를 추가하고, 동시 사건이 필요하면 softmax 대신 sigmoid multi-label로 전환한다.

## Codex에 줄 첫 요청

“`CODEX_CONTINUE.md`를 읽고 Raspberry Pi의 실제 MAD annotation 형식과 기존 체크포인트 구조를 검사한 뒤, dry-run 결과를 근거로 gunshot-only 추출과 4-class 재학습을 진행해줘. 기존 데이터는 삭제하지 마.”
