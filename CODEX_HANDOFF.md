# SAFIRA ESP32–Raspberry Pi 환경음 AI 인수인계

## 최종 목표

ESP32-WROOM-32D에 연결된 INMP441 마이크 3개의 소리를 Raspberry Pi로 전송하고, Raspberry Pi에서 총소리, 폭발, 사이렌, 화재경보, 유리 파손, 비명 같은 환경음을 AI 모델로 분류한다.

음성 문장을 받아 적는 작업이 아니라 **환경음 이벤트 분류**가 목적이다.

## 현재까지 확인된 상태

- ESP32 → Raspberry Pi 직렬 오디오 전송 성공
- Raspberry Pi에서 3채널 WAV 저장 성공
- 생성된 채널: `ambient_left`, `ambient_right`, `voice`
- 실제 생성 사례:
  - `esp32_20260812_162007_ambient_left.wav`
  - `esp32_20260812_162007_ambient_right.wav`
  - `esp32_20260812_162007_voice.wav`
- 환경음 판단에는 기본적으로 `ambient_left/right`를 사용한다.
- USB 재연결 시 포트가 `/dev/ttyUSB0`에서 `/dev/ttyUSB1` 등으로 바뀔 수 있다.

실제 녹음 WAV는 Raspberry Pi의 `/home/pi/joeun/recordings/`에 있으며 이 ZIP에는 포함되어 있지 않다.

## 오디오 및 직렬 프로토콜

- 16,000 Hz
- signed PCM16 little-endian
- 20 ms 프레임
- 채널당 320 samples
- 3채널 interleaved 순서: ambient left, ambient right, voice
- Raspberry Pi → ESP32 시작 명령: `START_INMP_ONLY\n`
- ESP32 응답: `STARTED\n`
- 마이크 패킷 헤더: `M3C0 + uint16 sequence + uint16 sample_count`
- 디버그 패킷: `DBG0`
- 스피커 역방향 패킷: `SPK0`

## 매우 중요한 설정

ESP32 펌웨어와 Raspberry Pi Python 코드의 `SERIAL_BAUD` 값은 반드시 같아야 한다.

실험 중 921600과 2000000을 모두 사용한 이력이 있으므로, ZIP의 값을 무조건 신뢰하지 말고 현재 보드에 마지막으로 업로드한 펌웨어 값과 맞춘다.

```bash
grep -R "SERIAL_BAUD" src tools receive_m3c0_wav.py 2>/dev/null
```

## Raspberry Pi 기본 실행

```bash
cd ~/joeun
source .venv/bin/activate
python3 -m serial.tools.list_ports
```

포트가 `/dev/ttyUSB1`인 예:

```bash
python3 receive_m3c0_wav.py \
  --port /dev/ttyUSB1 \
  --seconds 20 \
  --output-dir recordings
```

## ZIP 구성

- `esp32_firmware/`: PlatformIO ESP32 펌웨어 원본
- `raspberry_pi/`: 수신, WAV 저장, 재생 및 테스트 코드
- `docs/`: 프로토콜과 기존 인수인계 자료
- `AI_NEXT_STEP.md`: 환경음 AI 구현 요구사항

## 다음 Codex에게 요청할 일

1. 저장된 ambient WAV 파일을 사전학습 환경음 모델로 분류하는 독립 Python 프로그램 작성
2. Raspberry Pi에서 설치 가능한 가벼운 런타임 선택
3. 총소리 등 위험음 라벨을 한국어 결과로 매핑
4. 저장 WAV 검증 후 `M3C0` 실시간 수신 루프와 결합
5. 약 1초 슬라이딩 윈도우, 0.5초 간격 추론, 최근 점수 평균 적용
6. 콘솔과 JSON Lines 로그 출력
7. 임계값, 연속 감지 횟수, 직렬 포트, baud를 명령행 옵션으로 제공
8. 기존 수신 코드와 펌웨어 프로토콜을 깨뜨리지 말고 새 파일로 추가

## 안전 및 정확도 주의

- 실제 총기음을 큰 음량으로 재생하지 말고 공개된 테스트 WAV나 낮은 음량의 안전한 샘플로 검증한다.
- YAMNet 같은 범용 모델은 초기 검증용이다. 실제 제품 수준의 총소리 감지는 현장 데이터로 별도 평가와 전이학습이 필요하다.
- 단일 프레임 점수만으로 경보를 발생시키지 말고 시간 평균, 연속 감지, 최소 음량 조건을 적용한다.
