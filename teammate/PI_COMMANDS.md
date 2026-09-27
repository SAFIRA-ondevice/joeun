# 팀원용 Pi 설치·시연 명령어

실제 사용 버전은 live_all_mics_reenroll.py (좌우 주변 평균 + voice 참조). 수동 확인 후 [전원 자동 실행 설치](AUTOSTART.md)를 진행한다.

## 1. 노트북에서 Pi 접속

Pi와 같은 사용자 핫스팟에 연결한 뒤 터미널에서:

```bash
ssh pi@172.20.10.9
```

비밀번호는 `0000`. 입력 중 화면에 글자가 안 보이는 것은 정상이다. 접속 불가 시 Pi 전원/핫스팟을 확인하고 Pi 자체 터미널에서 `hostname -I`로 현재 IP를 확인한다. 이 주소는 다른 팀원 핫스팟에서 유지된다고 가정하지 않는다.

## 2. 기존 작업을 덮어쓰지 않는 별도 설치

실행 중 오디오 프로그램은 Ctrl+C로 종료하고 저장 완료를 기다린다. 다른 serial monitor도 닫는다. 아래 clone은 **처음 한 번** 실행한다. 폴더가 이미 있으면 삭제하지 말고 다음 업데이트 절차를 사용한다.

```bash
git clone --branch chatgpt/safira-audio-routing --single-branch https://github.com/SAFIRA-ondevice/joeun.git /home/pi/safira-runtime
cd /home/pi/safira-runtime/tactical_audio_ai_codex
```

이후 업데이트:

```bash
cd /home/pi/safira-runtime
git status --short
git pull --ff-only
cd tactical_audio_ai_codex
```

로컬 수정 때문에 pull이 거절되면 reset/clean으로 지우지 말고 담당자에게 변경 내용을 전달한다.

## 3. Python 환경 및 모델 준비

기존 Pi 환경이 남아 있으면 그대로 활성화한다:

```bash
source /home/pi/joeun-self-speech-test/tactical_audio_ai_codex/.venv/bin/activate
python -c "import torch, torchaudio, numpy, serial; print('의존성 OK', torch.__version__, torchaudio.__version__)"
```

기존 venv가 없는 새 64비트 Pi OS에서만:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv libsndfile1 git
cd /home/pi/safira-runtime/tactical_audio_ai_codex
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pyserial torch torchaudio
python -c "import torch, torchaudio, numpy, serial; print('의존성 OK', torch.__version__, torchaudio.__version__)"
```

Torch/torchaudio 설치 또는 공유 라이브러리 오류가 나면 `uname -m`, `python --version`, 오류 전문을 담당자에게 전달한다. 지원되는 wheel이 없는 환경에서는 위 설치가 성공한다고 보장하지 않는다. 동작 중인 기존 환경을 불필요하게 업그레이드하지 않는다.

모델은 Git에 올리지 않았다. 원래 Pi에 남아 있는 모델을 새 폴더로 복사한다:

```bash
cd /home/pi/safira-runtime/tactical_audio_ai_codex
mkdir -p models/separator_overlap_v1 models/classifier_noise_v1
cp -n /home/pi/joeun-self-speech-test/tactical_audio_ai_codex/models/separator_overlap_v1/best.pt models/separator_overlap_v1/best.pt
cp -n /home/pi/joeun-self-speech-test/tactical_audio_ai_codex/models/classifier_noise_v1/last.pt models/classifier_noise_v1/last.pt
ls -lh models/separator_overlap_v1/best.pt models/classifier_noise_v1/last.pt
```

파일이 없으면 모델 담당자에게 위 **두 checkpoint**를 받아 해당 경로에 넣어야 한다. 다른 classifier의 이름만 바꿔 넣지 않는다. `last.pt`가 최적 검증 모델이라는 뜻은 아니며 현재 전달된 시연 경로를 유지한 것이다.

## 4. 하드웨어 열기 전 모델 호환 검사

```bash
cd /home/pi/safira-runtime/tactical_audio_ai_codex
python - <<'PY'
import torch
from runtime_separator import Separator
from live_all_mics_reenroll import AmbientClassifier
s = torch.load('models/separator_overlap_v1/best.pt', map_location='cpu', weights_only=True)
assert s.get('task') == 'source_separation_v1', '분리 checkpoint 종류 불일치'
m = Separator().eval()
m.load_state_dict(s['state_dict'])
with torch.no_grad():
    stems, masks = m(torch.zeros(1, 16000))
assert tuple(stems.shape) == (1, 4, 16000)
assert torch.isfinite(stems).all()
c = AmbientClassifier('models/classifier_noise_v1/last.pt')
print('모델 로드 OK. 분리 품질 검증은 별도입니다.', c.classes)
PY
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

serial 권한 오류가 발생한 경우만 `sudo usermod -aG dialout pi` 실행 후 SSH를 완전히 종료하고 다시 로그인한다. 포트가 USB0가 아니라면 실행 명령의 --port를 실제 경로로 바꾼다.

## 5. 120초 시연 시작 — 터미널 A

```bash
cd /home/pi/safira-runtime/tactical_audio_ai_codex
python -u live_all_mics_reenroll.py --model models/separator_overlap_v1/best.pt --classifier-model models/classifier_noise_v1/last.pt --port /dev/ttyUSB0 --baud 1000000 --seconds 120 --speech-gain 1.5 --drone-gain 1.5 --gunshot-gain 0.15 --unknown-gain 1.5 --master 0.01 --combined-gain 1.0
```

처음 3초는 소리를 재생하거나 말하지 않는다. 프로그램은 등록 없이 시작한다. 총/드론 동시에 나와도 전체 감쇠하지 않고 각 추정 stem에 gain을 적용한다. Ctrl+C로 종료하면 녹음을 저장한다. 최대 --seconds 300이며 상시 서비스가 아니다. 녹음이 메모리에 쌓이므로 제한을 임의로 없애지 않는다.

## 6. 실행 중 5초 착용자 등록 — 터미널 B

다른 SSH 창에서:

```bash
ssh pi@172.20.10.9
cd /home/pi/safira-runtime/tactical_audio_ai_codex
python3 request_voice_enroll.py
```

터미널 A에서 `[재등록 시작]`이 뜨면 헤드셋을 착용한 본인만 5초 말한다. 주변 재생은 끈다. `[재등록 완료]`를 확인한다. 실패하면 기존 프로파일을 유지한다. 요청 스크립트는 모델/serial을 열지 않는다. 실행 중이 아닐 때 요청하면 파일이 남아 다음 실행 시 처리된다.

등록값을 초기화하려면 먼저 종료한 뒤 삭제 대신 보관한다:

```bash
cd /home/pi/safira-runtime/tactical_audio_ai_codex
if [ -f wearer_profile.json ]; then mv wearer_profile.json "wearer_profile.backup.$(date +%Y%m%d_%H%M%S).json"; fi
```

## 7. STARTED 또는 프레임 timeout

먼저 다른 프로그램이 포트를 사용 중인지 `fuser /dev/ttyUSB0`로 확인한다. 확인 없이 다른 프로세스를 강제 종료하지 않는다. ESP가 **이미 speaker 모드로 스트리밍 중임이 확인된 경우에만** 5번 명령 끝에 `--attach-speaker`를 붙인다. 이 옵션은 ESP를 speaker 모드로 바꾸는 명령이 아니다. 입력 전용 상태 또는 통신 단절은 ESP 담당자가 시작 모드/baud/배선을 확인해야 한다. Pi 전체 전원을 반복해서 껐다 켜는 것을 기본 복구 방법으로 삼지 않는다.

## 8. 결과 공유

프로그램 종료 시 출력한 결과 폴더를 먼저 확인한다:

```bash
cd /home/pi/safira-runtime/tactical_audio_ai_codex
ls -td recordings/live_all_mics_* | head -n 3
python3 -m http.server 8001 --directory recordings
```

같은 핫스팟 노트북에서 `http://172.20.10.9:8001/`에 접속해 해당 폴더를 선택한다. `headset_sent.wav`는 송신 PCM, `processed_before_master.wav`는 master 적용 전 진단 파일, `report.json`과 `self_metrics.json`은 상태 기록이다. gain 후 개별 stem은 진단용이며 실제 헤드셋 혼합·음소거 결과와 같지 않다. 공유 종료는 Ctrl+C. HTTP에는 인증이 없으므로 팀 네트워크에서 필요한 동안만 사용한다.

## 9. 팀 실기 체크

순서대로 무재생, 본인 발화, 외부 발화, 드론만, 총소리 녹음만, 총+드론, 본인+외부 발화를 기록한다. 등록 전후/재실행 후를 비교하고 각 실행의 모델 경로와 report를 보관한다. 실제 총기 대신 녹음 재생으로 시험한다. 출력 지연/클릭음은 저장 WAV와 실제 헤드셋 청취를 따로 기록한다. 이 시연은 청력 보호 장비 검증이 아니다.
