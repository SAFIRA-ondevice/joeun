# SAFIRA Raspberry Pi Audio Receiver

ESP32의 INMP441 마이크 3채널 스트림을 Raspberry Pi에서 수신하고 WAV로 검증하는 코드입니다.

## 스트림 규격

- USB serial: 2,000,000 baud
- 16,000 Hz, signed PCM16 little-endian
- 20 ms frame, 채널당 320 samples
- 채널 순서: ambient left, ambient right, voice
- packet header: M3C0 + uint16 sequence + uint16 sample count

ESP32 구현자는 [ESP32_RPI_AUDIO_PROTOCOL.md](ESP32_RPI_AUDIO_PROTOCOL.md)를 기준으로 값을 맞춰야 합니다.

## Raspberry Pi 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

ESP32가 연결된 포트를 확인합니다.

```bash
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

권한 오류가 발생하면 사용자를 dialout 그룹에 추가한 뒤 다시 로그인합니다.

```bash
sudo usermod -aG dialout "$USER"
```

## 10초 녹음

```bash
python3 receive_m3c0_wav.py \
  --port /dev/ttyUSB0 \
  --seconds 10 \
  --output-dir recordings
```

수신기는 ESP32에 `START_INMP_ONLY\n`을 전송하고 다음 세 파일을 생성합니다.

- `*_ambient_left.wav`
- `*_ambient_right.wav`
- `*_voice.wav`

정상 수신 기준은 `missing frames=0`, `bad headers=0`입니다.

## WAV 확인

```bash
file recordings/*.wav
aplay recordings/*_voice.wav
```

각 WAV는 16 kHz, mono, 16-bit PCM입니다.
