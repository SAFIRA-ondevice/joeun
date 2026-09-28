# AI 모델 평가 및 그래프 생성 가이드

> 목적: 현재 프로젝트의 **실제 저장 데이터/로그를 먼저 확인**하고,
> 가능한 경우 기존 모델을 그대로 평가하여 Confusion Matrix와 성능
> 그래프를 생성한다.\
> 데이터가 없으면 필요한 테스트만 새로 수행한다. **임의 수치나 가짜
> 그래프를 만들지 않는다.**
>
> 현재 확인된 주요 구성: - 분류 클래스: `speech`, `drone`, `gunshot` -
> Source separation 출력: `speech`, `drone`, `gunshot`, `background` -
> Separator: `models/separator_overlap_v1/best.pt` - 실험 classifier:
> `models/classifier_noise_v1/last.pt` - 기존 classifier 후보:
> `models/multilabel_10epoch.pt` - Runtime:
> `live_left_only_reenroll.py` - 사용자 등록 trigger:
> `request_voice_enroll.py`

------------------------------------------------------------------------

## 0. 먼저 지켜야 할 것

1.  기존 `.pt` 모델을 덮어쓰지 않는다.
2.  기존 학습 코드/실행 코드를 임의 수정하지 않는다.
3.  새 결과는 `evaluation_results/` 아래에 저장한다.
4.  학습 데이터 자체를 Test 데이터처럼 재사용해서 성능을 주장하지
    않는다.
5.  **Classifier가 single-label인지 multi-label인지 먼저 확인한다.**
6.  기존 평가 코드가 있으면 새 평가 코드를 만들기보다 기존 코드를 우선
    사용한다.
7.  데이터가 없으면 성능값을 추측해서 만들지 않는다.
8.  이번 작업은 기본적으로 **재학습이 아니라 기존 모델 평가**다.

------------------------------------------------------------------------

# 1. 프로젝트 위치 확인

프로젝트 경로를 알고 있다면 먼저 이동한다.

``` bash
cd /home/pi/joeun-self-speech-test/tactical_audio_ai_codex
pwd
ls -lah
```

경로가 다르면 프로젝트를 찾는다.

``` bash
find ~ -maxdepth 4 -type d -name "tactical_audio_ai_codex" 2>/dev/null
```

Python/가상환경 확인:

``` bash
python3 --version
ls -la .venv 2>/dev/null
```

가상환경이 있다면:

``` bash
source .venv/bin/activate
python --version
```

------------------------------------------------------------------------

# 2. 모델 파일 확인

``` bash
find . -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) -print
```

특히 다음 파일 확인:

``` bash
ls -lh models/separator_overlap_v1/best.pt 2>/dev/null
ls -lh models/classifier_noise_v1/last.pt 2>/dev/null
ls -lh models/multilabel_10epoch.pt 2>/dev/null
```

프로젝트 밖의 기존 모델도 확인하려면:

``` bash
find /home/pi -type f -name "multilabel_10epoch.pt" 2>/dev/null
```

**주의:** 여러 모델이 나오면 어떤 모델이 현재 시연에 사용되는지 실행
명령 또는 코드에서 확인한다.

``` bash
grep -RniE "classifier-model|multilabel_10epoch|classifier_noise|separator_overlap" . \
  --include="*.py" --include="*.sh" --include="*.md" 2>/dev/null
```

------------------------------------------------------------------------

# 3. Dataset / Validation / Test 데이터 확인

먼저 디렉터리 이름으로 찾는다.

``` bash
find . -type d \( \
  -iname "*train*" -o \
  -iname "*val*" -o \
  -iname "*valid*" -o \
  -iname "*test*" -o \
  -iname "*dataset*" -o \
  -iname "*data*" \
\) -print
```

오디오 데이터 확인:

``` bash
find . -type f \( \
  -iname "*.wav" -o \
  -iname "*.flac" -o \
  -iname "*.mp3" \
\) | head -200
```

전체 오디오 개수:

``` bash
find . -type f \( -iname "*.wav" -o -iname "*.flac" -o -iname "*.mp3" \) | wc -l
```

라벨/메타데이터 후보 확인:

``` bash
find . -type f \( \
  -iname "*.csv" -o \
  -iname "*.json" -o \
  -iname "*.jsonl" -o \
  -iname "*.txt" \
\) -print
```

클래스명이 파일/코드에 어떻게 기록되어 있는지 확인:

``` bash
grep -RniE "speech|drone|gunshot|background" . \
  --include="*.py" --include="*.csv" --include="*.json" --include="*.txt" \
  2>/dev/null | head -200
```

------------------------------------------------------------------------

# 4. 가장 중요: Classifier가 Single-label인지 Multi-label인지 확인

`multilabel_10epoch.pt`라는 이름 때문에 **multi-label 모델일 가능성이
있으므로 반드시 확인**한다.

학습/평가 코드 검색:

``` bash
find . -type f -name "*.py" -print | sort
```

다음 키워드 확인:

``` bash
grep -RniE "BCEWithLogitsLoss|CrossEntropyLoss|sigmoid|softmax|multilabel|multi_label|MultiLabel" . \
  --include="*.py" 2>/dev/null
```

### 판별 기준

-   `CrossEntropyLoss` + `softmax` + 정답 하나 → 보통 **single-label**
-   `BCEWithLogitsLoss` + `sigmoid` + 클래스별 0/1 → 보통
    **multi-label**

최종 판단은 실제 학습 코드의 **target 생성 방식과 loss**를 같이 보고
한다.

### 왜 중요한가?

#### Single-label이면

일반적인 하나의 3×3 Confusion Matrix:

``` text
             Pred
           S   D   G
Actual S
       D
       G
```

#### Multi-label이면

하나의 3×3 행렬을 억지로 만들면 안 된다.

대신 각 클래스별 2×2 Confusion Matrix:

``` text
Speech   : TN FP / FN TP
Drone    : TN FP / FN TP
Gunshot  : TN FP / FN TP
```

그리고 다음을 같이 제시한다.

-   Precision
-   Recall
-   F1-score
-   Micro average
-   Macro average

------------------------------------------------------------------------

# 5. 기존 Evaluation 코드부터 확인

``` bash
find . -type f \( \
  -iname "*eval*.py" -o \
  -iname "*test*.py" -o \
  -iname "*metric*.py" -o \
  -iname "*valid*.py" \
\) -print
```

평가 관련 함수 검색:

``` bash
grep -RniE "confusion_matrix|classification_report|precision|recall|f1|accuracy|evaluate|validation" . \
  --include="*.py" 2>/dev/null
```

**기존 평가 코드가 발견되면 그 코드가 모델을 어떻게 로드하고 Dataset을
어떻게 읽는지 먼저 확인한다.**

새로운 loader를 임의로 만들지 않는다.

------------------------------------------------------------------------

# 6. 기존 학습 로그 확인

CSV/JSON 로그:

``` bash
find . -type f \( \
  -iname "*history*.csv" -o \
  -iname "*history*.json" -o \
  -iname "*log*.csv" -o \
  -iname "*metrics*.csv" -o \
  -iname "*result*.csv" \
\) -print
```

TensorBoard:

``` bash
find . -type f -name "events.out.tfevents.*" -print
```

로그 디렉터리 후보:

``` bash
find . -type d \( \
  -iname "runs" -o \
  -iname "logs" -o \
  -iname "*tensorboard*" -o \
  -iname "*experiment*" \
\) -print
```

학습 코드가 loss를 어디에 저장했는지 검색:

``` bash
grep -RniE "train_loss|val_loss|valid_loss|accuracy|history|SummaryWriter|add_scalar" . \
  --include="*.py" 2>/dev/null
```

------------------------------------------------------------------------

# 7. 기존 결과 이미지/CSV 확인

``` bash
find . -type f \( \
  -iname "*confusion*" -o \
  -iname "*matrix*" -o \
  -iname "*loss*.png" -o \
  -iname "*accuracy*.png" -o \
  -iname "*metric*.png" -o \
  -iname "*report*.txt" -o \
  -iname "*report*.csv" \
\) -print
```

이미 만들어 둔 결과가 있다면 **어떤 모델/데이터로 만든 것인지 확인한
뒤** 재사용한다.

------------------------------------------------------------------------

# 8. 여기까지 확인한 뒤 판단

  -----------------------------------------------------------------------
  발견된 것                           해야 할 일
  ----------------------------------- -----------------------------------
  Test/Validation 데이터 + 평가 코드  현재 모델로 평가만 다시 실행
  있음                                

  Test/Validation 데이터만 있음       기존 Dataset/model loading 방식을
                                      이용해 평가 스크립트 작성

  학습 history 있음                   Loss/Accuracy 그래프 생성

  TensorBoard 로그 있음               TensorBoard에서 학습 곡선 확인/추출

  기존 confusion matrix 있음          사용 모델/데이터 일치 여부 확인 후
                                      사용

  `.pt` 모델만 있고 Test 데이터 없음  새 독립 Test 데이터 수집 필요

  학습 데이터만 있음                  학습 데이터 성능을 최종 Test
                                      성능처럼 사용하지 말 것

  Source separation 평가 코드 있음    코드에서 실제 지원하는 분리 지표로
                                      평가

  Wearer suppression 결과 없음        별도의 Before/After 실험 수행
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 9. 결과 폴더 생성

``` bash
mkdir -p evaluation_results/classifier
mkdir -p evaluation_results/training
mkdir -p evaluation_results/separation
mkdir -p evaluation_results/wearer_suppression
```

권장 구조:

``` text
evaluation_results/
├── classifier/
│   ├── confusion_matrix.png
│   ├── confusion_matrix.csv
│   ├── classification_report.txt
│   └── predictions.csv
├── training/
│   ├── loss_curve.png
│   └── accuracy_curve.png
├── separation/
│   ├── separation_metrics.csv
│   └── separation_metrics.png
└── wearer_suppression/
    ├── measurements.csv
    ├── wearer_before_after.png
    └── external_preservation.png
```

------------------------------------------------------------------------

# 10. Classifier Confusion Matrix 생성

## 10-1. 먼저 실제 inference 코드를 찾는다

``` bash
grep -RniE "torch.load|load_state_dict|classifier|predict|forward" . \
  --include="*.py" 2>/dev/null | head -300
```

**현재 프로젝트의 model class / feature extraction / preprocessing을
그대로 사용해야 한다.**

오디오를 그냥 다른 방식으로 전처리하면 실제 모델 평가가 달라질 수 있다.

------------------------------------------------------------------------

## 10-2. Single-label인 경우 기본 평가 구조

아래 코드는 **완성본이 아니라 구조 템플릿**이다.

`TODO`는 반드시 기존 프로젝트 코드에 맞게 연결한다.

파일 예:

``` bash
nano evaluate_classifier.py
```

``` python
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

RESULT_DIR = Path("evaluation_results/classifier")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["speech", "drone", "gunshot"]

# ---------------------------------------------------------
# TODO 1:
# 프로젝트 기존 코드와 동일한 방식으로 모델을 로드한다.
# 예:
# model = ...
# state = torch.load(...)
# model.load_state_dict(...)
# model.eval()
#
# TODO 2:
# 기존 validation/test Dataset을 그대로 로드한다.
#
# TODO 3:
# 학습 당시와 동일한 preprocessing / feature extraction을 사용한다.
# ---------------------------------------------------------

y_true = []
y_pred = []

# TODO:
# for sample, label in test_loader:
#     prediction = ...
#     y_true.append(...)
#     y_pred.append(...)

if not y_true:
    raise RuntimeError(
        "평가 데이터가 연결되지 않았습니다. "
        "기존 Dataset/model loading 코드를 먼저 확인하세요."
    )

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
)

np.savetxt(
    RESULT_DIR / "confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d",
)

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
)

(RESULT_DIR / "classification_report.txt").write_text(
    report,
    encoding="utf-8",
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
)

fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, values_format="d")
ax.set_title("Classifier Confusion Matrix")
fig.tight_layout()
fig.savefig(
    RESULT_DIR / "confusion_matrix.png",
    dpi=200,
)
plt.close(fig)

with open(
    RESULT_DIR / "predictions.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow(["actual", "predicted"])

    for actual, predicted in zip(y_true, y_pred):
        writer.writerow([
            CLASS_NAMES[actual],
            CLASS_NAMES[predicted],
        ])

print(report)
print("Saved:", RESULT_DIR)
```

필요 패키지 확인:

``` bash
python -c "import sklearn, matplotlib, numpy; print('OK')"
```

없다면 프로젝트 환경을 먼저 확인하고 필요한 경우에만 설치한다.

------------------------------------------------------------------------

# 11. Multi-label Classifier인 경우

Multi-label이면 일반 3×3 confusion matrix 대신:

``` python
from sklearn.metrics import multilabel_confusion_matrix
```

을 사용한다.

개념:

``` python
cms = multilabel_confusion_matrix(y_true, y_pred)
```

`y_true`, `y_pred` 예:

``` text
[speech, drone, gunshot]

[1, 0, 1]
[0, 1, 0]
[1, 1, 0]
```

각 클래스마다:

``` text
[[TN, FP],
 [FN, TP]]
```

가 생성된다.

Classification report:

``` python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["speech", "drone", "gunshot"],
        digits=4,
        zero_division=0,
    )
)
```

**중요:** sigmoid threshold도 학습/기존 inference 코드에서 사용한 값을
그대로 사용한다.

예를 들어 기존 코드가 `0.5`가 아닌 threshold를 쓰고 있다면 임의로 0.5로
바꾸지 않는다.

------------------------------------------------------------------------

# 12. Train / Validation Loss 그래프

## 로그가 CSV로 남아 있는 경우

먼저 컬럼 확인:

``` bash
head -10 경로/history.csv
```

또는:

``` bash
python - <<'PY'
import pandas as pd

p = "경로/history.csv"
df = pd.read_csv(p)

print(df.columns.tolist())
print(df.head())
PY
```

예를 들어 실제 컬럼이 다음과 같다면:

``` text
epoch,train_loss,val_loss,train_accuracy,val_accuracy
```

다음처럼 그린다.

``` bash
nano plot_training_history.py
```

``` python
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "경로/history.csv"

OUT = Path("evaluation_results/training")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

print(df.columns.tolist())

required = ["epoch", "train_loss", "val_loss"]

for col in required:
    if col not in df.columns:
        raise RuntimeError(f"CSV에 {col} 컬럼이 없습니다.")

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(df["epoch"], df["train_loss"], label="Train Loss")
ax.plot(df["epoch"], df["val_loss"], label="Validation Loss")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training / Validation Loss")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "loss_curve.png", dpi=200)
plt.close(fig)

if (
    "train_accuracy" in df.columns
    and "val_accuracy" in df.columns
):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        df["epoch"],
        df["train_accuracy"],
        label="Train Accuracy",
    )
    ax.plot(
        df["epoch"],
        df["val_accuracy"],
        label="Validation Accuracy",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training / Validation Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "accuracy_curve.png", dpi=200)
    plt.close(fig)

print("Saved:", OUT)
```

**컬럼명이 다르면 실제 CSV 컬럼명에 맞춰 변경한다.**

------------------------------------------------------------------------

# 13. TensorBoard 로그만 있는 경우

설치 여부:

``` bash
tensorboard --version
```

실행:

``` bash
tensorboard --logdir ./runs --host 0.0.0.0 --port 6006
```

`runs`가 아니라면 앞에서 찾은 실제 로그 경로를 넣는다.

확인할 항목:

-   Train Loss
-   Validation Loss
-   Train Accuracy
-   Validation Accuracy
-   학습 epoch에 따른 수렴 여부

TensorBoard에 없는 값은 새로 만들어내지 않는다.

------------------------------------------------------------------------

# 14. Source Separation 평가

Source separation은 `speech / drone / gunshot / background` 중 하나를
고르는 단순 분류가 아니다.

따라서 **Classifier처럼 일반 Confusion Matrix를 만드는 것이 기본
평가법은 아니다.**

먼저 기존 코드가 어떤 metric을 지원하는지 찾는다.

``` bash
grep -RniE "SI-SDR|SI_SDR|SDR|SNR|PESQ|STOI|metric|separation" . \
  --include="*.py" 2>/dev/null
```

평가 코드 검색:

``` bash
find . -type f \( \
  -iname "*separat*eval*.py" -o \
  -iname "*eval*separat*.py" -o \
  -iname "*metric*.py" \
\) -print
```

### 원칙

기존 코드에서 실제로 지원하는 metric을 우선 사용한다.

예:

-   SI-SDR
-   SDR
-   SNR

**프로젝트가 계산하지 않던 지표를 발표를 위해 임의로 추가해서 기존
성능인 것처럼 표현하지 않는다.**

분리 평가를 하려면 일반적으로 다음이 필요하다.

``` text
혼합 음원
+
정답 Speech stem
+
정답 Drone stem
+
정답 Gunshot stem
+
정답 Background stem
```

즉 Ground Truth stem이 없다면 일부 객관적 separation metric은 계산할 수
없다.

------------------------------------------------------------------------

# 15. Wearer Self-Speech Suppression 평가

이번 Runtime Wearer Enrollment 기능은 Confusion Matrix보다 **Before /
After 비교**가 적절하다.

목표:

> 사용자 등록 후 착용자의 자기 음성은 감소하면서 외부 사람의 음성은
> 가능한 한 유지되는지 확인

최소 다음 **3개 조건**을 테스트한다.

### Test A - Wearer only

착용자만 동일 문장을 말한다.

``` text
A_before.wav
A_after.wav
```

### Test B - External speaker only

외부 사람만 동일 위치/거리에서 말한다.

``` text
B_before.wav
B_after.wav
```

### Test C - Wearer + External simultaneous

착용자와 외부 사람이 동시에 말한다.

``` text
C_before.wav
C_after.wav
```

------------------------------------------------------------------------

# 16. Wearer 테스트 순서

가능하면 마이크 위치, 볼륨, 거리 등을 바꾸지 않는다.

## Before

아직 해당 사용자의 profile을 적용하지 않은 조건에서:

``` text
A_before
B_before
C_before
```

녹음.

## Enrollment

Runtime AI 실행 상태에서:

1.  등록 요청
2.  시작 비프
3.  약 5초 동안 착용자가 말함
4.  profile 생성
5.  완료 비프
6.  `[재등록 완료]` 확인

## After

같은 조건에서:

``` text
A_after
B_after
C_after
```

다시 녹음.

가능하면 Before/After에서 **같은 문장**을 사용한다.

------------------------------------------------------------------------

# 17. Wearer 억제 정량화

가장 간단하게 RMS와 dBFS를 비교할 수 있다.

``` python
import wave
import numpy as np


def load_wav(path):
    with wave.open(path, "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        frames = w.readframes(w.getnframes())

    if width != 2:
        raise ValueError("현재 예시는 PCM16 WAV 기준입니다.")

    x = np.frombuffer(frames, dtype=np.int16)
    x = x.astype(np.float32) / 32768.0

    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)

    return x


def rms(x):
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def dbfs(x):
    return float(20 * np.log10(rms(x) + 1e-12))


before = load_wav("A_before.wav")
after = load_wav("A_after.wav")

before_db = dbfs(before)
after_db = dbfs(after)

suppression_db = before_db - after_db

print("Before:", before_db, "dBFS")
print("After :", after_db, "dBFS")
print("Suppression:", suppression_db, "dB")
```

단, 녹음 앞뒤의 무음 길이가 크게 다르면 전체 RMS 비교가 왜곡될 수 있다.

가능하면 **동일 구간 길이 / 동일 발화 구간**을 비교한다.

------------------------------------------------------------------------

# 18. Wearer 결과에서 봐야 하는 것

### A. Wearer only

등록 후 출력 레벨이 의미 있게 감소했는지 확인.

``` text
Wearer Before → Wearer After 감소
```

### B. External only

외부 사람의 음성이 과도하게 감소하면 안 된다.

``` text
External Before ≈ External After
```

### C. Wearer + External

착용자 음성은 감소하지만 외부 음성은 최대한 유지되는지 확인.

이 세 조건이 있어야 단순히

> "전체 Speech Gain을 낮춰서 조용해진 것"

과

> "착용자 음성을 선택적으로 억제한 것"

을 구분하기 쉽다.

------------------------------------------------------------------------

# 19. 기존 Test 데이터가 없다면 새로 필요한 Classifier 테스트

Classifier 평가 데이터가 전혀 없다면 **독립 Test set**을 만든다.

최소 클래스:

``` text
speech
drone
gunshot
```

가능하면 클래스별 샘플 수를 비슷하게 맞춘다.

예:

``` text
test_classifier/
├── speech/
├── drone/
└── gunshot/
```

단, 모델이 **multi-label**이라면 이 폴더 구조만으로 부족할 수 있다.

예:

``` text
speech + drone
speech + gunshot
drone + gunshot
speech + drone + gunshot
```

같은 중첩 상황도 정답 라벨을 가질 수 있기 때문이다.

따라서 먼저 학습 코드에서 target 구조를 확인한다.

------------------------------------------------------------------------

# 20. 새 Test 데이터 수집 시 조건

가능하면 다음을 포함한다.

-   조용한 환경
-   실제 사용 환경의 background noise
-   가까운 음원
-   먼 음원
-   서로 다른 크기의 음원
-   실제 장비/마이크로 녹음한 데이터

중요:

### 학습 데이터와 Test 데이터 분리

학습 때 사용한 동일 파일을 Test에 다시 넣으면 성능이 실제보다 좋아 보일
수 있다.

가능하면:

``` text
Train: 학습에 사용
Validation: 학습 중 모델 선택
Test: 최종 평가에만 사용
```

으로 분리한다.

------------------------------------------------------------------------

# 21. 새 Test 데이터가 필요한 경우의 우선순위

시간이 부족하면 다음 순서로 한다.

### 1순위 - Classifier

정답 라벨이 있는 독립 테스트 음원 확보.

결과:

-   Confusion Matrix 또는 class별 confusion matrices
-   Precision
-   Recall
-   F1
-   Accuracy (적절한 경우)

### 2순위 - Wearer suppression

``` text
Wearer only
External only
Wearer + External
```

Before/After 실험.

### 3순위 - Source separation

Ground Truth stem까지 확보할 수 있을 때 객관적 separation metric 평가.

------------------------------------------------------------------------

# 22. 최종적으로 발표자료에 우선 넣을 그래프

## 1. Classifier Confusion Matrix

가장 우선.

단:

-   Single-label → 하나의 일반 Confusion Matrix
-   Multi-label → 클래스별 2×2 Confusion Matrix

로 구분한다.

## 2. Train / Validation Curve

학습 로그가 **실제로 남아 있을 때만** 사용.

추천:

``` text
Epoch vs Train Loss
Epoch vs Validation Loss
```

Accuracy 로그도 있다면 추가 가능.

## 3. Wearer Voice Suppression Before / After

예:

``` text
Wearer only     Before ██████████
                After  ██

External voice  Before ████████
                After  ███████
```

이 그래프는 이번 Runtime Enrollment 기능의 효과를 설명하는 데 사용한다.

------------------------------------------------------------------------

# 23. 최종 결과 확인 명령

``` bash
find evaluation_results -type f -maxdepth 3 -print
```

PNG:

``` bash
find evaluation_results -type f -name "*.png" -print
```

CSV:

``` bash
find evaluation_results -type f -name "*.csv" -print
```

TXT:

``` bash
find evaluation_results -type f -name "*.txt" -print
```

------------------------------------------------------------------------

# 24. 팀원이 최종적으로 기록해야 할 정보

평가 결과만 보내지 말고 아래 정보도 같이 기록한다.

``` text
[Classifier]
사용 모델:
평가 데이터:
Test sample 수:
Single-label / Multi-label:
Threshold:
Accuracy:
Precision:
Recall:
F1:

[Source Separation]
사용 모델:
평가 데이터:
사용 metric:
결과:

[Wearer Suppression]
등록 시간:
Wearer-only suppression:
External speech before:
External speech after:
Wearer+External 결과:
```

모델 이름과 Test set을 기록하지 않으면 나중에 어떤 실험 결과인지
구분하기 어렵다.

------------------------------------------------------------------------

# 25. 최종 체크리스트

-   [ ] 현재 사용 classifier 모델 확인
-   [ ] 현재 사용 separator 모델 확인
-   [ ] Train / Validation / Test 데이터 위치 확인
-   [ ] Classifier target encoding 확인
-   [ ] Single-label / Multi-label 확인
-   [ ] 기존 evaluation 코드 확인
-   [ ] 기존 학습 history 확인
-   [ ] TensorBoard log 확인
-   [ ] 기존 평가 결과 확인
-   [ ] Test 데이터가 있으면 기존 모델 평가
-   [ ] Confusion Matrix 생성
-   [ ] Precision / Recall / F1 생성
-   [ ] 학습 로그가 있을 때만 Loss/Accuracy curve 생성
-   [ ] Source separation 기존 metric 확인
-   [ ] Wearer Before/After 실험
-   [ ] 결과를 `evaluation_results/`에 저장
-   [ ] 사용 모델과 Test dataset 기록
-   [ ] 기존 `.pt` 파일 변경하지 않음
-   [ ] 기존 Runtime 코드 변경하지 않음
-   [ ] 임의 수치/가짜 그래프 사용하지 않음

------------------------------------------------------------------------

# 핵심 요약

가장 먼저 **기존 Test/Validation 데이터, 학습 로그, 평가 코드가 남아
있는지 확인**한다.

있다면 새 학습이나 새 데이터 수집 없이 현재 모델을 다시 평가해서 결과를
생성한다.

없다면 필요한 것만 추가한다.

1.  **Classifier Test set 없음** → 독립적인 정답 라벨 Test set 추가 수집
2.  **Training log 없음** → 과거 Train/Val loss 그래프는 복원 불가능할
    수 있음. 임의 생성 금지
3.  **Source separation Ground Truth 없음** → 객관적 separation metric
    계산 가능 여부부터 확인
4.  **Wearer suppression 결과 없음** → Wearer / External / Simultaneous
    3조건 Before/After 테스트

최종 발표용 우선 결과는 다음 3개다.

1.  **Classifier Confusion Matrix + Precision/Recall/F1**
2.  **Train/Validation Loss Curve (기존 로그가 있을 경우)**
3.  **Wearer Voice Suppression Before/After 비교**
