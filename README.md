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
| 전공 용어집 | 물리학 용어 28개 및 번역 출력 검증 구현 |
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

응답에는 `translation`, `prompt_tokens`, `generated_tokens`, `elapsed_seconds`가 포함됩니다. `context`에는 앞 영어 원문을 전달하며 마지막 256자를 사용합니다. `generated_tokens`는 성공한 마지막 시도의 실제 생성 토큰 수이며 재시도 합계는 시간 로그에서 확인합니다. 입력과 출력의 합이 KoreanLM의 2,048토큰 제한을 넘으면 HTTP 422를 반환합니다. M4·16GB에서 메모리 충돌을 피하기 위해 번역 요청은 한 번에 하나씩 실행됩니다.

### 영어 STT

```http
POST /api/v1/transcriptions?language=en
Content-Type: multipart/form-data
```

```bash
curl -X POST 'http://localhost:8000/api/v1/transcriptions?language=en' \
  -F 'file=@MIT8_01F16_W06PS01-2_360p.mp4'
```

STT가 끝나면 인접한 영어 조각을 문장 단위로 묶어 KoreanLM으로 번역합니다. 응답은 `segments`만 반환하며, 각 묶음은 첫 조각의 시작·마지막 조각의 종료 시각과 영문·한글 자막을 포함합니다. 원래 STT 구간 수와 반환 구간 수는 다를 수 있습니다.

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

문장 끝의 `.?!`, 18초·350자 상한, 2초를 초과하는 구간 사이 공백에서 묶음을 닫습니다. 원본 조각을 분할하지 않으므로 단일 조각이 상한을 넘으면 그대로 보존합니다. 단어별 정렬이나 원래 조각에 대한 번역 재배분은 제공하지 않습니다.

앞 묶음의 영어 원문 마지막 256자를 다음 문맥으로 사용하며, 번역 실패 후에도 해당 원문 문맥을 유지합니다. 원문에 등장하는 물리학 용어를 `data/physics_glossary.json`에서 찾아 번역 지시에 포함합니다. 한글·용어·명시적 숫자·지원 수식 기호·단위를 검증합니다. 수식의 `d v r`/`dvR` 같은 표기 별칭과 `dmRu`/`dmR u`, 독립 변수를 보호하고 수학 문맥의 prime 표기도 확인합니다. 반복·일부 지시문 유출·손상 문자를 검출하면 생성을 조기 종료합니다. 토큰 상한에서 끝난 출력도 실패로 처리합니다. 실패한 출력은 문맥 없이 한 번 재시도하며, 두 시도 모두 실패하면 `translation=null`로 반환합니다. 이 규칙은 의미 정확성을 보장하지 않으며 수식 관계·부정·바꿔 쓴 반복은 별도 평가가 필요합니다.

지원 확장자는 MP4, MOV, MKV, WEBM, WAV, MP3, M4A입니다. 업로드 파일은 임시 파일로 처리한 뒤 요청 종료 시 삭제합니다. STT와 번역 요청은 모델별로 한 번에 하나씩 실행됩니다.

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

각 구간에는 위의 다섯 필드가 모두 포함됩니다. 구간 번호는 배열 인덱스로 식별합니다. 성공·부분 실패·음성 없음은 모두 HTTP 200입니다. 모든 구간의 번역 검증이 실패해도 영어 원문과 타임스탬프를 반환합니다. `translation_failed`는 토큰 제한 초과 또는 재시도 후 출력·용어·문맥 반복 검증 실패를 의미합니다. 검증을 통과한 번역도 의미 정확성을 보장하지 않으므로 품질 평가는 별도로 수행합니다.

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

현재 API는 원본 모델을 4bit로 변환해 사용하며 LoRA 어댑터를 적용하지 않습니다. [공식 추론 예시](https://github.com/quantumaikr/KoreanLM#추론)의 base+LoRA 구성과 다릅니다. 다만 2026-10-05 공식 고정 리비전의 실제 파일을 확인한 결과 B 행렬 64개가 모두 0이어서 해당 어댑터는 학습된 변화를 추가하지 않습니다. 다운로드·MLX 변환은 완료했으며 전체 모델 추론은 하지 않았습니다. [어댑터 점검 결과](evaluation/lora_conversion.md)에 근거와 재현 방법을 기록했습니다.

## 다음 작업

1. Qwen3 후보에 물리학 용어집·수식 보존 지시를 적용해 평가 문장 22개를 재검토합니다. 두 후보의 기본 실행 비교에서 의미 보존은 TranslateGemma 8/22·Qwen3 15/22였으며 둘 다 주요 오역이 남아 API 모델은 유지했습니다. [실행 결과와 후속 작업](evaluation/candidate-comparison.md)을 참고하세요.
2. 부정·수식 연산 관계·문장 간 의미 반복을 판정하는 평가를 확대하고, STT 문장 경계·전문 용어 인식과 용어 별칭을 개선합니다.
3. 생성·재시도 로그를 기준으로 출력 한도·재시도 정책·프롬프트 캐시·재처리 캐시의 시간과 품질을 비교합니다.
4. 작업 ID·진행률·백그라운드 처리와 결과 스트리밍을 구현하고 프론트엔드 처리 상태 UI를 연결합니다.
5. SRT/VTT 생성과 영상 자막 재생 API를 추가하고 다른 전공 평가 데이터로 확대합니다.

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

모델 생성은 테스트 대역으로 검증하며, 실제 미디어 디코더로 손상된 파일을 확인합니다. 문장 묶기·타임스탬프·직전 영어 문맥 전달, 번역 재시도·숫자·수식·단위·스트림 조기 종료와 자원 정리를 검증합니다. 실제 모델의 번역 품질 평가는 별도로 수행합니다.

## 번역 품질 평가

물리학 용어집 28개, 합성 평가 문장 12개, MIT 공식 자막과 수동 한국어 기준 문장 10개를 제공합니다. 평가는 API와 같은 모델·프롬프트·재시도 로직을 사용합니다.

```bash
.venv-api/bin/python scripts/evaluate_translation.py --output evaluation/physics-current.json
.venv-api/bin/python scripts/evaluate_translation.py \
  --media MIT8_01F16_W06PS01-2_360p.mp4 --output evaluation/lecture-current.json
.venv-api/bin/python scripts/evaluate_lecture_quality.py --output evaluation/lecture-quality.json
.venv-api/bin/python scripts/evaluate_lecture_quality.py \
  --source stt --output evaluation/lecture-quality-current.json
```

`evaluate_lecture_quality.py`는 저장된 `lecture.json`의 STT와 공식 자막을 별도로 번역하며 STT를 재실행하지 않습니다. `--source stt`는 저장 STT만 번역하고 `--source official`은 공식 원문만 번역합니다. `--cases-only`는 수동 기준 문장에 한정하고 `--stt-only`는 모델 없이 공식 자막 대조만 수행합니다. 공식 자막의 수식 오기도 보존하므로 WER은 표기 차이의 대리지표로만 해석합니다.

개선 전 구현은 `git show 40bed6d:app/services/translation.py`를 별도 Python 파일로 저장하고 `--service-file`로 전달해 비교할 수 있습니다. 과거 조각별 한국어 문맥 실행은 `--raw-segments --context-mode previous_translation`으로 지정합니다. 평가 결과에는 번역문, 기준 번역, 용어 포함 여부, 금지어, 실패 및 처리 시간이 기록됩니다. 모델 생성 실패도 결과에 포함합니다.

용어 통과율은 번역 정확도가 아닙니다. 관계를 뒤집거나 부정을 추가한 문장은 용어가 있어도 오역입니다. 문맥 복사 검증은 공백을 제외한 완전 일치만 탐지하며, 일부 표현 반복이나 바꿔 쓴 문맥 복사는 탐지하지 않습니다. 원문이 실제로 반복된 경우에도 동일 번역을 거부할 수 있습니다. 지정 용어의 동의어 역시 검증 실패가 될 수 있어, 정확한 의미 검증과 유연한 용어 별칭 처리는 후속 작업입니다.

상세 결과와 한계는 [번역 품질 평가 보고서](evaluation/README.md)를 확인하세요.

## 작업 시간 로그

`make api`의 Uvicorn INFO 로그에 요청·STT·번역 시간을 기록합니다. 수정 후 실행 중인 서버를 종료하고 `make api`로 다시 시작해야 적용됩니다. Python 단독 실행에서 같은 로그를 보려면 `logging.basicConfig(level=logging.INFO)`로 로깅을 활성화합니다.

각 API 요청은 `request_id`로 구분하며, 구간별 로그에는 0부터 시작하는 `segment_index`가 들어갑니다. 해당 식별자는 스레드풀의 서비스 로그에도 전달됩니다. 다음은 형식 예시이며 측정 결과가 아닙니다.

```text
INFO: event=request_started request_id=abc123 segment_index=None elapsed_seconds=0.000 method=POST path=/api/v1/transcriptions
INFO: event=stt request_id=abc123 segment_index=None elapsed_seconds=20.100 status=ok segment_count=45
INFO: event=translation_generate request_id=abc123 segment_index=0 elapsed_seconds=3.200 status=ok attempt=1
INFO: event=translation_attempt request_id=abc123 segment_index=0 elapsed_seconds=3.210 status=ok attempt=1 prompt_tokens=240 max_tokens=128 raw_output_tokens=30 accepted=True
INFO: event=translation_total request_id=abc123 segment_index=0 elapsed_seconds=3.220 status=ok attempts=1
INFO: event=sentence_grouping request_id=abc123 segment_index=None elapsed_seconds=0.001 status=ok source_segment_count=45 sentence_count=36
INFO: event=translation_batch request_id=abc123 segment_index=None elapsed_seconds=220.000 status=ok source_segment_count=45 sentence_count=36 failed_segments=1
INFO: event=http_request request_id=abc123 segment_index=None elapsed_seconds=240.500 method=POST path=/api/v1/transcriptions status_code=200 status=ok error_type=None
```

| 로그 이벤트 | 측정 범위 |
| --- | --- |
| `http_request` | API 요청의 본문 수신·검증·처리·응답 전송. 실패 응답도 포함 |
| `upload_save` | multipart 파싱 후 업로드 파일을 임시 파일에 복사 |
| `stt` | 스레드풀 대기와 디코딩·모델 준비·STT 전체 |
| `stt_decode` | 오디오 디코딩 및 샘플 검증 |
| `stt_model_load` | 최초 Whisper 모델 로딩 |
| `stt_queue` | STT 추론 잠금 대기 |
| `stt_inference` | 음성 인식과 지연 구간 generator 순회 |
| `translation_model_load` | 서버 시작 시 KoreanLM 모듈·모델·샘플러 로딩 |
| `sentence_grouping` | STT 조각을 문장으로 묶는 시간과 원본·묶음 개수 |
| `translation_batch` | 전체 구간 번역과 실패 구간 수 |
| `segment_translation` | 한 구간의 스레드풀 대기와 번역 처리 |
| `translation_queue` | 번역 모델 잠금 대기 |
| `translation_generate` | 각 시도의 모델 생성 시간 |
| `translation_attempt` | 프롬프트 구성·생성·검증, 토큰 수·종료 이유·검증 사유·통과 여부 |
| `translation_total` | 한 번역 호출의 대기·재시도 전체와 시도 횟수 |

시간 단위는 초이며 시스템 시각 변경에 영향받지 않는 `perf_counter`로 측정합니다. 상위 이벤트는 하위 이벤트를 포함하므로 모든 로그 시간을 더하면 중복 계산됩니다. `accepted=False`는 생성은 끝났지만 검증에 실패해 재시도하거나 해당 구간이 실패한다는 의미입니다. 예외·취소 시 `status=error`, `error_type`을 기록하며 원래 예외를 전달합니다. HTTP 200의 부분 실패는 `translation_batch.failed_segments`로 구분합니다.

`raw_output_tokens`는 각 시도의 실제 MLX 생성 토큰 수입니다. `finish_reason`은 정상 종료 `stop`, 토큰 상한 `length`, 조기 중단 시 `None`일 수 있습니다. `early_stop_reason`은 `repetition`, `prompt_leak`, `invalid_character` 등 조기 중단 이유이고, `rejection_reason`에는 검증 실패 이유도 기록됩니다. 번역문·프롬프트·업로드 파일 내용은 시간 로그에 넣지 않습니다. 서버 시작 시 모델 로그와 API 외의 직접 서비스 호출은 `request_id=-`입니다.

## 수행 시간 개선 후보

기존 전체 영상 평가의 단계 시간 합은 246.46초이고 번역이 221.71초로 약 90%입니다. 먼저 새 로그의 생성·재시도·대기 시간을 확인하고 같은 영상에서 의미 정확도와 함께 비교합니다.

1. 문장 묶기·영어 문맥·반복 중단을 구현했습니다. 같은 영상의 번역 시간과 의미 오류·실패 구간 수를 함께 비교합니다.
2. 원문 길이에 맞는 출력 한도와 반복 중단을 비교합니다. 고정 128토큰을 무조건 낮추면 정상 번역도 잘릴 수 있습니다.
3. 주변 영어 원문 문맥을 사용해 오역 전파와 재시도를 줄이는지 확인합니다. 프롬프트 길이는 제한합니다.
4. 같은 영상 재처리 결과를 캐시합니다. 첫 실행 시간에는 효과가 없으며 모델·프롬프트·STT 설정 변경 시 캐시를 무효화해야 합니다.
5. 공통 프롬프트 캐시·배치 추론은 현재 MLX 버전과 16GB 메모리 범위에서 별도 검증합니다. 직전 한국어 번역에 의존하는 구조에서 단순 병렬 호출은 잠금 때문에 직렬 실행됩니다.

STT를 더 작은 모델이나 낮은 beam으로 실행하는 방법은 전문 용어 인식 품질과 교환 관계가 있고, 전체 시간의 주된 병목인 번역에는 영향이 없습니다. 작업 ID·스트리밍은 결과를 빨리 보여주는 방법이며 총 추론 시간이 줄어드는지는 별도로 측정합니다. 모델 구성·교체는 품질과 속도를 함께 논의한 후 진행합니다.
