SAFIRA AI 감지 결과 서버 연동 규격 v1

라즈베리파이에서 환경음 AI 추론을 수행하고, 서버에는 아래 JSON 형식으로 감지 결과를 전송하도록 구현합니다. 아래 규격은 기존 AI 출력 구조를 기준으로 정한 서버 연동 규격입니다.

**1. 전송 방식**

- 방향: 라즈베리파이 → 서버
- 방식: HTTPS POST `/api/v1/audio-results`
- 데이터 형식: `application/json`
- 전송 시점: AI 추론 결과가 생성될 때마다 전송
- 기본 추론 간격 설정: 250ms. 실제 전송 간격은 처리 시간에 따라 달라질 수 있음
- 서버 수신 대상: AI 분석 결과. 원본 음성 전송은 이 API에 포함하지 않음

**2. 전송 예시**

아래 수치는 형식 설명을 위한 예시입니다.

```json
{
  "schema_version": "1.0",
  "message_id": "550e8400-e29b-41d4-a716-446655440000",
  "device_id": "SAFIRA_001",
  "timestamp": "2026-09-17T07:20:10.250Z",
  "payload": {
    "targets": {
      "speech": 12.3,
      "drone": 86.4,
      "gunshot": 72.1
    },
    "detected": ["drone", "gunshot"],
    "fallback": null,
    "final": "DRONE + GUNSHOT",
    "inference_ms": 35.2
  }
}
```

**3. 필드 정의**

| 변수명 | 자료형 / 범위 | 의미 |
|---|---|---|
| `schema_version` | string, `"1.0"` | 데이터 규격 버전 |
| `message_id` | string, UUID | 결과별 고유 ID. 재전송 시 동일 ID 유지 |
| `device_id` | string | 헤드셋 장치 고유 ID |
| `timestamp` | string, UTC ISO 8601 | AI 결과 생성 시각 |
| `payload.targets.speech` | number, 0~100 | 사람 음성 감지 점수 |
| `payload.targets.drone` | number, 0~100 | 드론 감지 점수 |
| `payload.targets.gunshot` | number, 0~100 | 총성 감지 점수 |
| `payload.detected` | string[] | 장치의 판정 기준을 넘은 클래스 목록. 미검출 시 `[]` |
| `payload.fallback` | object 또는 null | 보조 환경음 모델 결과. 실행하지 않았으면 `null` |
| `payload.final` | string | 화면 표시용 최종 판정 문구 |
| `payload.inference_ms` | number, 0 이상 | 추론 및 판정 처리 시간, 밀리초 |

위 필드는 모두 포함합니다. `fallback`은 값으로 `null`을 허용합니다.

**4. 서버 처리 규칙**

- 세 감지 점수는 각각 독립적이므로 합이 100일 필요가 없습니다.
- 여러 소리가 동시에 감지될 수 있으므로 `detected`는 배열로 처리합니다.
- 감지 여부는 `detected`를 기준으로 처리합니다. `final` 문자열을 분석해서 판단하지 않습니다.
- `speech`는 사람 목소리 감지이며, 음성 인식 텍스트가 아닙니다.
- `UNKNOWN`은 소리를 분류하지 못했다는 의미입니다. 무음이나 안전 상태를 뜻하지 않습니다.
- 동일 `message_id`가 재전송되면 중복 저장하지 않습니다.

**5. 보조 환경음 결과**

주요 클래스가 감지되지 않았고 YAMNet 보조 모델을 실행한 경우, `payload.fallback`은 다음 구조입니다.

```json
{
  "recognized": true,
  "label": "Siren",
  "confidence": 0.72,
  "top": [
    {"label": "Siren", "score": 0.72},
    {"label": "Vehicle", "score": 0.18},
    {"label": "Music", "score": 0.05}
  ]
}
```

- `recognized`: boolean, 보조 모델의 판정 기준 충족 여부
- `label`: string, 추정 소리 이름. 기준 미충족 시 `"UNKNOWN"`
- `confidence`: number, 0~1
- `top`: 상위 후보 배열. 각 항목은 `label`과 0~1 범위의 `score` 포함
- `targets`는 0~100이고, 보조 모델의 `confidence`와 `score`는 0~1이므로 단위를 구분합니다.

**6. 서버 응답**

정상 수신 및 저장 완료 시 HTTP 200과 아래 응답을 반환합니다.

```json
{
  "message_id": "550e8400-e29b-41d4-a716-446655440000",
  "accepted": true
}
```
