# 전원 자동 실행 설치

현재 실제 사용 파일은 `live_all_mics_reenroll.py`다. L/R 평균 + voice 참조를 사용한다. 기존 왼쪽 전용 버전은 당시 입력 문제의 비교·복구용으로 유지한다. 배선 접합 문제가 관여했을 가능성이 있다는 사용자 보고이며 수리 후 실기 확인은 필요하다.

먼저 PI_COMMANDS.md의 환경·모델 준비를 마치고 수동 명령에서 파일 이름을 `live_all_mics_reenroll.py`로 바꿔 실제 출력을 확인한다. 아래 명령은 Pi SSH 터미널에서 실행한다. 사용자 pi, 설치 폴더 /home/pi/safira-runtime 기준이다.

## 1. 설치 준비

수동 오디오 프로그램을 Ctrl+C로 종료하고 serial monitor를 닫는다.

```bash
cd /home/pi/safira-runtime
git pull --ff-only
cd tactical_audio_ai_codex
python3 -m py_compile live_all_mics_reenroll.py runtime_separator.py request_voice_enroll.py
sudo usermod -aG dialout pi
sudo install -m 644 deploy/safira-audio.env /etc/default/safira-audio
sudo nano /etc/default/safira-audio
```

환경 파일 install은 최초 설치 때만 한다. SAFIRA_PYTHON은 torch가 설치된 Python 절대 경로다. 기존 venv 사용 시 기본값을 유지하고 새 venv 사용 시 다음으로 변경한다:

```text
SAFIRA_PYTHON=/home/pi/safira-runtime/tactical_audio_ai_codex/.venv/bin/python
```

SAFIRA_MODEL/SAFIRA_CLASSIFIER는 실제 모델 경로, SAFIRA_PORT는 ESP 포트다. `ls -l /dev/serial/by-id/`로 확인한 고정 장치 경로를 써도 된다. SAFIRA_ATTACH=0은 냉간 부팅 기본값이다. 저장: Ctrl+O, Enter, Ctrl+X.

## 2. 서비스 설치

```bash
cd /home/pi/safira-runtime/tactical_audio_ai_codex
sudo install -m 644 deploy/safira-audio.service /etc/systemd/system/safira-audio.service
sudo systemd-analyze verify /etc/systemd/system/safira-audio.service
sudo systemctl daemon-reload
sudo systemctl enable --now safira-audio.service
systemctl status safira-audio.service --no-pager
journalctl -u safira-audio.service -n 60 --no-pager
```

active(running)만으로 출력 준비 완료는 아니다. 잡음 보정 완료, 반복 추론 로그와 실제 헤드셋 출력을 확인한다. 부팅→장치 대기→모델 로드→ESP 시작→3초 잡음 보정 후 동작한다. 전원 인가 즉시 출력은 아니다. 보정 중에는 시험 소리를 틀지 않는다.

서비스는 --seconds 0으로 지속 실행한다. WAV 및 전체 추론 기록을 누적하지 않아 메모리 증가를 방지하고 종료 시 recordings/service_status에 최신 상태를 덮어쓴다. 일반 시연은 --seconds 120으로 녹음한다. 설치 후 장시간 메모리 상태와 냉간 부팅은 팀원이 확인해야 한다.

## 3. 실행 중 재등록

```bash
cd /home/pi/safira-runtime/tactical_audio_ai_codex
python3 request_voice_enroll.py
journalctl -u safira-audio.service -f
```

재등록 시작 로그 후 본인만 5초 말한다. 일반 재생은 등록 중 잠시 음소거된다. 완료 후 정상 처리로 돌아오며 프로파일은 다음 실행에서도 사용한다. 로그 창 Ctrl+C는 서비스를 종료하지 않는다. L/R로 바꾸었으므로 이전 왼쪽용 프로파일은 새로 등록한다. GPIO 버튼은 아직 구현되지 않았다.

## 4. 재부팅 확인

작업을 저장한 뒤:

```bash
sudo reboot
```

SSH가 끊어진 뒤 다시 접속해서 확인한다:

```bash
systemctl is-enabled safira-audio.service
systemctl status safira-audio.service --no-pager
journalctl -u safira-audio.service -b -n 80 --no-pager
```

오디오 계산에는 핫스팟/인터넷이 필요하지 않고 SSH 관리에만 필요하다. 전원을 끌 때는 `sudo poweroff` 후 분리한다.

## 5. 운영 명령

중지: `sudo systemctl stop safira-audio.service`

시작: `sudo systemctl start safira-audio.service`

재시작: `sudo systemctl restart safira-audio.service`

자동 실행 해제 및 중지: `sudo systemctl disable --now safira-audio.service`

수동 녹음을 하려면 먼저 서비스를 stop하고 PI_COMMANDS.md 시연 명령을 실행한다. 종료 후 서비스를 start한다. serial 소유 프로그램을 동시에 두 개 켜지 않는다.

## 6. 업데이트

```bash
sudo systemctl stop safira-audio.service
cd /home/pi/safira-runtime
git status --short
git pull --ff-only
cd tactical_audio_ai_codex
sudo install -m 644 deploy/safira-audio.service /etc/systemd/system/safira-audio.service
sudo systemctl daemon-reload
sudo systemctl start safira-audio.service
journalctl -u safira-audio.service -n 60 --no-pager
```

모델/프로파일/환경 파일은 덮어쓰지 않는다. Git 충돌 시 reset/clean으로 지우지 않는다.

## 7. 오류 처리

- Python/모델 없음: /etc/default/safira-audio 경로를 확인한다. 모델은 Git에 없다.
- 장치 없음: 포트·배선·dialout 권한 확인. 장치를 최대 30초 기다리고 오류 종료 후 10초마다 재시도한다.
- STARTED ACK 없음: 서비스를 stop하고 ESP 담당자가 시작 명령/현재 모드를 확인한다. 이미 speaker 모드라는 확인이 있을 때만 SAFIRA_ATTACH=1로 설정한다. 입력 전용 모드를 speaker로 바꾸는 옵션이 아니다. 냉간 부팅도 다시 시험해야 한다.
- ESP 담당자는 이미 실행 중이어도 START_INMP_SPEAKER에 STARTED를 다시 응답하는지 확인한다. 서비스 재시도가 펌웨어 상태 복구를 대신하지는 않는다.
- 탁탁 소리: 서비스 설치가 해결하는 문제는 아니다. 원음/송신 WAV와 실제 헤드셋을 비교하고 전원·접지·I2S 경로를 확인한다.
- 좌우 평균 후 소리가 약하면 반대 극성에 의한 상쇄 가능성을 포함해 배선과 각 채널을 확인한다.

systemd 서비스 및 Bash 실행 검증은 Pi의 위 명령으로 수행한다. 이번 업로드에서 실기 자동 부팅 성공을 주장하지 않는다.
