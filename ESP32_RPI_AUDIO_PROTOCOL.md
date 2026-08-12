# ESP32 ↔ Raspberry Pi 오디오 프로토콜

이 문서는 SAFIRA ESP32 마이크 펌웨어와 Raspberry Pi 수신 코드가 공유하는 기준 규격이다.

## 전송 설정

| 항목 | 값 |
|---|---|
| Transport | USB serial |
| Baud rate | 2,000,000 |
| Sample rate | 16,000 Hz |
| PCM | signed 16-bit, little-endian |
| Frame duration | 20 ms |
| Samples per channel | 320 |
| Input channels | 3 |

`921600 baud`에서는 3채널 16 kHz PCM16 데이터가 시리얼 실효 전송량을 초과하므로 2,000,000 baud를 사용한다.

## 스트림 시작

Raspberry Pi가 다음 ASCII 명령을 보낸다.

```text
START_INMP_ONLY\n
```

ESP32는 다음 응답 후 binary packet 전송을 시작한다.

```text
STARTED\n
```

## 공통 헤더

Python `struct` 표현은 `<4sHH`이며 전체 8 bytes다.

| Offset | Size | Type | 설명 |
|---:|---:|---|---|
| 0 | 4 | char[4] | packet magic |
| 4 | 2 | uint16 LE | sequence, 65535 다음 0 |
| 6 | 2 | uint16 LE | sample count 또는 payload byte count |

## M3C0 마이크 패킷

- Magic: `M3C0`
- Header count: `320`
- Payload: 1,920 bytes
- 전체 packet: 1,928 bytes
- 전송 주기: 초당 50 packet

Payload는 frame-interleaved PCM16이다.

```text
ambient_left[0], ambient_right[0], voice[0],
ambient_left[1], ambient_right[1], voice[1],
...
ambient_left[319], ambient_right[319], voice[319]
```

## DBG0 디버그 패킷

- Magic: `DBG0`
- Header count: `24` (sample count가 아니라 payload bytes)
- Payload: `<IIIIII`

Payload 순서:

1. ESP32 uptime milliseconds
2. microphone frames sent
3. speaker packets received
4. speaker frames written
5. voice frames read
6. free heap bytes

## SPK0 스피커 패킷

- Magic: `SPK0`
- Header count: `320`
- Payload: mono PCM16 640 bytes

ESP32는 mono 샘플을 stereo 좌우로 복제하여 PCM5102A로 출력한다.

## ESP32 기준 상수

```cpp
static constexpr uint32_t SERIAL_BAUD = 2000000;
static constexpr int SAMPLE_RATE_HZ = 16000;
static constexpr size_t FRAME_SAMPLES = 320;
static constexpr size_t INPUT_CHANNELS = 3;
```

## 핀 기준

| 장치 | BCLK | WS/LRCLK | DATA |
|---|---:|---:|---:|
| Ambient INMP441 L/R | GPIO26 | GPIO25 | GPIO34 input |
| Voice INMP441 | GPIO27 | GPIO14 | GPIO35 input |
| PCM5102A | GPIO27 | GPIO14 | GPIO22 output |

INMP441 SEL 설정:

- Ambient L: GND
- Ambient R: 3V3
- Voice: GND

## 호환성 확인

ESP32와 Raspberry Pi에서 다음 값이 모두 같아야 한다.

- baud rate
- sample rate
- frame samples
- channel count와 순서
- signed PCM16 little-endian
- `<4sHH` header layout
