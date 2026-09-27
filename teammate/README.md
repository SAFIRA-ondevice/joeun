# SAFIRA 팀 인계 — 실행 중 착용자 재등록

첨부된 COMPLETE 실행 파일을 검토하여 반영했다. 실기 동작 기록은 작업자 보고이며 이번 업로드 후 Pi 재검증은 아직 하지 않았다. **팀원은 [설치부터 시연까지 명령어](PI_COMMANDS.md)를 먼저 따른다.** 기존 raspberry_pi 아래 프로토타입 설명과 최신 실행 경로를 구분한다.

## Pi 접속

- 사용자 핫스팟에서 IP: `172.20.10.9` (네트워크 변경 또는 DHCP 재할당 시 확인 필요)
- 호스트 이름 및 SSH 계정: `pi`
- 비밀번호: `0000`
- 접속: `ssh pi@172.20.10.9`
- 프로젝트: `/home/pi/joeun-self-speech-test/tactical_audio_ai_codex`
- 공개 저장소에 위 접속 정보를 포함하는 것은 소유자가 명시적으로 요청했다.

## 변경된 동작

- 시작 시 3초 잡음 보정 후 AI 실행. 시작 시 음성 등록을 강제하지 않는다.
- 저장된 `wearer_profile.json`이 있으면 로드한다. 프로파일이 없을 때 자기 음성 억제가 오작동하지 않아야 한다.
- 별도 터미널에서 `python request_voice_enroll.py` 실행 시 `.reenroll_request`로 재등록 요청.
- 주 실행 프로그램이 요청을 받아 시작 알림음, 안정화 대기, 5초 음성 수집, 프로파일 검증 및 저장, 완료 알림음을 수행한다.
- 성공 시 새 프로파일을 적용하며 프로세스를 재시작하지 않는다. 실패 시 기존 프로파일을 유지한다.
- 이는 음향 참조 보정이다. 개인별 voiceprint 학습이나 완전한 본인/외부 화자 분리를 의미하지 않는다.

## 실행

```bash
cd /home/pi/joeun-self-speech-test/tactical_audio_ai_codex
source .venv/bin/activate
python -u live_left_only_reenroll.py --model models/separator_overlap_v1/best.pt --classifier-model models/classifier_noise_v1/last.pt --seconds 120 --speech-gain 1.5 --drone-gain 1.5 --gunshot-gain 0.15 --unknown-gain 1.5 --master 0.01 --combined-gain 1.0
```

재등록 요청은 동일 프로젝트 폴더의 다른 터미널에서 실행한다.

```bash
cd /home/pi/joeun-self-speech-test/tactical_audio_ai_codex
source .venv/bin/activate
python request_voice_enroll.py
```

모델 가중치, 녹음, 개인 프로파일, 임시 요청 파일, 백업은 이번 코드 업로드 대상이 아니다. 위 모델 경로는 Pi에서 사용 중이라고 전달된 경로이며 모델 품질을 보증하지 않는다.

## 적용 정책 및 검증 범위

외부 말소리·드론·모든 미분류 소리 ×1.5, 총소리 ×0.15, 출력 master ×0.01. 동시에 총/드론이 감지되어도 전체 혼합을 감쇠하지 않는 설정은 `--combined-gain 1.0`이다. 분리 모델의 누출 때문에 실제 소리별 감쇠량은 설정값과 다를 수 있다.

주변 입력은 왼쪽 채널, 착용자 참조는 voice 채널을 사용한다. 오른쪽 채널 제외는 관찰된 입력 문제를 피하기 위한 조치다. 자기 음성 억제로 외부 소리까지 제거되는지는 동시 발화로 검증해야 한다.

작업자가 보고한 확인: 프로파일 없이 실행, 요청 수신, 5초 재등록 완료, AI 재시작 없이 계속 처리. 남은 확인: 저장 프로파일 재시작 로드, 실패 시 이전 값 유지, 본인/외부/동시 발화, 총/드론 동시 재생, 실제 귀까지 지연 측정.

담당별 내용: [ESP32](ESP32.md), [서버](SERVER.md).

## 이번 업로드의 범위와 코드 보완

핵심 코드 세 파일: `tactical_audio_ai_codex/live_left_only_reenroll.py`, `request_voice_enroll.py`, `runtime_separator.py`. 전달된 COMPLETE 파일은 첫 번째 이름으로 통일했다. runtime_separator는 기존 source_separation.py의 Separator 정의만 추출한 것으로 학습 도구를 포함하지 않는다. 기존 training/audio_model.py를 그대로 참조한다. 가중치와 아키텍처 호환은 Pi 명령어의 사전 검사로 확인한다.

프로파일/요청 경로를 실행 파일 기준으로 맞추고, 저장 성공 후에만 새 프로파일을 적용한다. 잘못된 저장 프로파일은 무시하고 자기 억제 없이 실행한다. 재등록 시작 이전의 추론 결과를 폐기하며 알림음에도 master를 적용한다. 기본 combined-gain은 1.0으로 변경했다. 설정을 명시한 아래 명령을 권장한다.

등록 중 일반 오디오 출력은 음소거된다. 등록 동안 AI가 무중단으로 소리를 재생한다는 뜻은 아니다. 수신과 주 프로세스는 유지하며 등록용 추론을 수행한다. 5초 수집 외에도 시작 비프, 안정화 250 ms, 검증용 문맥 수집 시간이 추가된다. 완료 두 번 비프, 실패 세 번 비프이며 master가 작으면 작게 들릴 수 있다.

자기 음성 후보에서 **전체 출력 음소거**가 남아 있어 동시에 말하는 외부 사람이나 드론도 사라질 수 있다. 이는 자기 음성만 완전히 분리한 구현이 아니다. unknown/background ×1.5에는 잡음도 포함된다. 분류 점수는 정확도 확률 보증이 아니며 무음 때 드론 오탐은 해결 완료가 아니다.

로컬 확인: Python 문법, 합성 PCM 패킷/프레이밍, 프로파일 유효성/실패, gain·mute 정책 검사. Torch 모델 실행, Pi 실기·실제 등록·클릭 제거·음원 분리 품질은 이번 환경에서 검증하지 못했다.
