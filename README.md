# K-aption BE

해외 대학 강의의 영어 음성을 인식하고 강의 문맥을 반영한 한국어 번역을 제공하는 FastAPI 백엔드입니다.

- 저장소: https://github.com/Raising-Lee-Hyun-seong/K-aption_BE
- STT: faster-whisper (`transcribe.py` 공통 구현, `requirements-stt.txt` 의존성)
- 번역: `quantumaikr/KoreanLM`의 MLX 4bit 변환본
- 실행 장비: Apple Silicon Mac

## 현재 범위

| 기능 | 상태 |
| --- | --- |
| 영상·음성 업로드 및 영어 STT | FastAPI 구현 |
| 문맥을 포함한 한국어 번역 | FastAPI 구현 |
| KoreanLM 4bit 변환 | 구현 및 M4·16GB 실행 확인 |
| 전공 용어집 | 후속 작업 |
| SRT/VTT 생성 및 자막 재생 | 후속 작업 |

STT 구현은 프로젝트 내부 `transcribe.py`에 있으며 API가 이를 공유합니다. `requirements-api.txt`는 `requirements-stt.txt`를 포함해 STT 의존성을 설치합니다. STT와 번역 결과를 터미널에 반환하는 CLI는 사용하지 않습니다. 프론트엔드는 HTTP API의 JSON 응답을 사용합니다.

## 실행 준비

프로젝트 루트에서 실행합니다. Python 3.9 이상과 Apple Silicon Mac이 필요합니다.

```bash
git clone https://github.com/Raising-Lee-Hyun-seong/K-aption_BE.git
cd K-aption_BE

make setup-api
make setup-mlx
make model
make api
```

이미 `models/koreanlm-4bit`이 준비된 장비에서는 `make model`을 다시 실행하지 않습니다.

```bash
make setup-api       # FastAPI·STT·번역 환경 설치
make setup-mlx       # KoreanLM 변환 환경 설치
make model           # 원본 다운로드 → FP16 → MLX 4bit
make api             # 기본 주소: http://localhost:8000
```

- Swagger UI: http://localhost:8000/swagger
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json
- 상태 확인: http://localhost:8000/health

## API

### 상태 확인

```http
GET /health
```

```json
{
  "status": "ok",
  "translation_model_loaded": true,
  "transcription_model_loaded": false
}
```

KoreanLM은 서버 시작 시 한 번 로드됩니다. faster-whisper는 첫 STT 요청에서 로드한 후 재사용하므로, STT 실행 전에는 `transcription_model_loaded`가 `false`입니다.

### 한국어 번역

```http
POST /api/v1/translations
Content-Type: application/json
```

```json
{
  "text": "The velocity is constant.",
  "context": "We are discussing uniform motion.",
  "max_tokens": 128
}
```

```bash
curl -X POST http://localhost:8000/api/v1/translations \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "The velocity is constant.",
    "context": "We are discussing uniform motion.",
    "max_tokens": 128
  }'
```

응답에는 `translation`, `prompt_tokens`, `generated_tokens`, `elapsed_seconds`가 포함됩니다. 입력과 출력의 합이 KoreanLM의 2,048토큰 제한을 넘으면 HTTP 422를 반환합니다. M4·16GB에서 메모리 충돌을 피하기 위해 번역 요청은 한 번에 하나씩 실행됩니다.

### 영어 STT

```http
POST /api/v1/transcriptions?language=en
Content-Type: multipart/form-data
```

```bash
curl -X POST 'http://localhost:8000/api/v1/transcriptions?language=en' \
  -F 'file=@MIT8_01F16_W06PS01-2_360p.mp4'
```

STT가 끝나면 각 영어 구간을 KoreanLM으로 바로 번역합니다. 응답은 프론트엔드가 자막으로 바로 사용할 수 있도록 `segments`만 반환하며, 각 구간은 시작·종료 시각과 영문·한글 자막을 포함합니다.

```json
{
  "segments": [
    {
      "start": 3.6,
      "end": 7.44,
      "text": "The velocity is constant.",
      "translation": "속도는 일정합니다.",
      "translation_error": null
    }
  ]
}
```

각 구간을 번역할 때 직전 구간의 한국어 번역을 문맥으로 사용합니다. 번역 검증 실패 시 해당 구간의 원문을 보존하고 다음 구간의 문맥을 초기화합니다. API는 번역 결과에서 프롬프트 표식을 제거하며, 한글이 없는 결과는 문맥 없이 한 번 더 생성합니다. 지원 확장자는 MP4, MOV, MKV, WEBM, WAV, MP3, M4A입니다. 업로드 파일은 임시 파일로 처리한 뒤 요청 종료 시 삭제합니다. STT와 번역 요청은 모델별로 한 번에 하나씩 실행됩니다.

Apple Silicon에서 faster-whisper의 mel 계산 중 허위 overflow 경고가 발생하지 않도록 API 환경은 NumPy 1.26 계열을 사용합니다. `make setup-api`가 해당 버전을 설치합니다. 디코딩 결과가 비어 있거나 `NaN`·`inf`를 포함하면 추론 전에 HTTP 422로 거부합니다.

## 자막 API 계약

`POST /api/v1/transcriptions`는 `multipart/form-data`의 필수 `file` 필드로 영상·음성을 받습니다. `language`는 생략하거나 `en`으로 지정합니다. 다른 언어는 HTTP 422로 거부합니다. 처리가 끝날 때까지 연결을 유지하는 동기 응답이며, 현재 작업 ID와 진행률 조회는 제공하지 않습니다.

| 필드 | 형식 | 의미 |
| --- | --- | --- |
| `segments` | 배열 | 인식 순서의 구간 목록. 음성이 없으면 빈 배열 |
| `start` | 숫자 | 영상 시작 기준 시작 시각(초), 0 이상 |
| `end` | 숫자 | 종료 시각(초), `start`보다 크며 디코딩된 오디오 길이 이하 |
| `text` | 문자열 | 영어 원문 |
| `translation` | 문자열 또는 null | 한국어 번역. 번역 검증 실패 시 null |
| `translation_error` | 문자열 또는 null | 성공 시 null, 검증 실패 시 `translation_failed` |

각 구간에는 위의 다섯 필드가 모두 포함됩니다. 구간 번호는 배열 인덱스로 식별합니다. 성공·부분 실패·음성 없음은 모두 HTTP 200입니다. 모든 구간의 번역 검증이 실패해도 영어 원문과 타임스탬프를 반환합니다. `translation_failed`는 토큰 제한 초과 또는 재시도 후 출력 검증 실패를 의미합니다. 검증을 통과한 번역도 의미 정확성을 보장하지 않으므로 품질 평가는 별도로 수행합니다.

부분 실패 응답:

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 3.0,
      "text": "The velocity is constant.",
      "translation": "속도는 일정합니다.",
      "translation_error": null
    },
    {
      "start": 3.0,
      "end": 6.0,
      "text": "The net force is zero.",
      "translation": null,
      "translation_error": "translation_failed"
    }
  ]
}
```

음성 없음 응답:

```json
{"segments": []}
```

### 오류와 프론트엔드 처리

| HTTP 상태 | 조건 | 프론트엔드 처리 |
| --- | --- | --- |
| 200 | 전체 성공 또는 번역 검증 부분 실패 | 구간별 `translation_error` 확인. 실패 구간은 영어 원문과 번역 실패 표시로 재생 |
| 200 | `segments`가 빈 배열 | 인식된 음성이 없음을 표시 |
| 415 | 지원하지 않는 확장자 | 지원 확장자를 안내하고 파일 재선택 |
| 422 | 필수 파일 누락, 허용하지 않는 언어, 오디오 디코딩·검증 실패 | 입력 오류를 표시하고 파일·요청 확인 |
| 500 | 예상하지 못한 추론·서버 오류 | 처리 실패를 표시. 자동 재시도는 하지 않음 |

미디어 오류의 `detail`은 문자열입니다.

```json
{"detail": "The uploaded media does not contain decodable audio"}
```

FastAPI 요청 형식 검증 오류의 `detail`은 배열입니다.

```json
{
  "detail": [
    {"type": "missing", "loc": ["body", "file"], "msg": "Field required", "input": null}
  ]
}
```

500 응답은 JSON 형식을 보장하지 않습니다. 프론트엔드는 상태 코드를 먼저 확인하고 JSON이 아니면 일반 처리 실패 메시지를 표시합니다. 번역 단독 API `POST /api/v1/translations`는 부분 결과 없이 토큰 제한·출력 검증 실패를 HTTP 422로 반환합니다.

## 프론트엔드 연결

기본 CORS 허용 origin은 `http://localhost:3000`, `http://localhost:5173`입니다. 다른 주소를 사용할 때는 쉼표로 구분합니다.

```bash
KAPTION_CORS_ORIGINS='http://localhost:3000,http://localhost:5173' make api
```

| 환경 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `KAPTION_MODEL_DIR` | `models/koreanlm-4bit` | MLX 4bit 모델 경로 |
| `KAPTION_STT_MODEL` | `small` | faster-whisper 모델 이름 |
| `KAPTION_STT_DEVICE` | `cpu` | STT 실행 장치 |
| `KAPTION_STT_COMPUTE_TYPE` | `int8` | STT 연산 형식 |
| `KAPTION_CORS_ORIGINS` | 로컬 프론트엔드 2개 | 허용 origin 목록 |

브라우저에서는 `FormData`를 사용하고 `Content-Type`을 직접 설정하지 않습니다. 브라우저가 multipart 경계값을 설정합니다.

```javascript
/** 영상을 업로드하고 전체·부분 실패 및 음성 없음 결과를 구분한다. */
async function transcribeVideo(file) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("http://localhost:8000/api/v1/transcriptions?language=en", {
    method: "POST",
    body: form,
  });
  const raw = await response.text();
  let payload;
  try {
    payload = JSON.parse(raw);
  } catch {
    throw new Error(`영상 처리 실패 (HTTP ${response.status})`);
  }
  if (!response.ok) {
    const detail = payload.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg).join(", ")
      : typeof detail === "string" ? detail : "영상 처리 실패";
    throw new Error(message);
  }
  return {
    segments: payload.segments,
    hasSpeech: payload.segments.length > 0,
    failedTranslations: payload.segments.filter(
      (segment) => segment.translation_error !== null
    ).length,
  };
}
```

Uvicorn worker를 여러 개 실행하면 worker마다 KoreanLM을 중복 로드합니다. 현재 M4·16GB 환경에서는 worker 하나를 사용합니다.

## KoreanLM 4bit 모델 준비

원본 KoreanLM은 PyTorch `.bin` 형식입니다. `scripts/prepare_koreanlm.py`가 고정된 원본 리비전을 다운로드하고 FP16 safetensors로 변환한 뒤 MLX에서 4bit로 양자화합니다.

```bash
make setup-mlx
make model
```

- 원본 모델: `quantumaikr/KoreanLM`
- 고정 리비전: `f4351abcdd6a933afbaffad0badf60c273e71920`
- FP16 중간 모델: `models/koreanlm-fp16`
- 최종 모델: `models/koreanlm-4bit`
- 원본 캐시: `.cache/huggingface`

모델, 캐시, 가상환경, 업로드 영상과 실행 결과는 Git에서 제외합니다.

## M4·16GB 검증 결과

4bit 모델 크기는 약 3.5GiB입니다. 짧은 문장 세 번을 실행했을 때 모델 로딩은 1.73~1.99초, 생성 속도는 초당 18~22토큰, MLX 최대 메모리는 4.23~4.41GB였습니다.

실행 가능성은 확인했지만 번역 품질 기준은 통과하지 않았습니다. `The net force acting on the object is zero.`를 `물체의 중력이 없습니다.`로 번역하는 의미 오류가 있었습니다. 프롬프트, 문맥 구성과 전공 용어집을 개선하고 5~10분 강의 구간으로 다시 평가해야 합니다.

## 다음 작업

1. 프론트엔드에서 영상 업로드와 처리 상태 UI를 연결합니다.
2. 긴 작업을 요청·작업 ID 기반 백그라운드 처리로 전환합니다.
3. 긴 영상의 STT·번역을 요청·작업 ID 기반 스트리밍 처리로 전환합니다.
4. 전공 용어집 20개와 번역 품질 평가 데이터를 준비합니다.
5. SRT/VTT 생성과 영상 자막 재생 API를 추가합니다.

## 참고 자료

- [KoreanLM 원본 코드](https://github.com/quantumaikr/KoreanLM)
- [KoreanLM 모델 카드](https://huggingface.co/quantumaikr/KoreanLM)
- [KoreanLM 모델 설정](https://huggingface.co/quantumaikr/KoreanLM/blob/main/config.json)
- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)

KoreanLM 기반 코드의 라이선스는 저장소의 `LICENSE`를 확인하세요. 강의 영상과 생성 자막의 이용 조건은 코드와 별도로 관리합니다.

## 통합 회귀 테스트

```bash
.venv-api/bin/python -m unittest discover -s tests -v
```

모델 생성은 테스트 대역으로 검증하며, 실제 미디어 디코더로 손상된 파일을 확인합니다. 구간 순서·타임스탬프·직전 번역 문맥 전달, 번역 재시도와 토큰 제한, 오류 시 업로드 파일 종료 및 임시 파일 삭제를 검증합니다. 실제 모델의 번역 품질 평가는 별도로 수행합니다.
