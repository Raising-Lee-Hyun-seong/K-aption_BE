# K-aption BE

해외 대학 강의의 영어 음성을 인식하고 강의 문맥을 반영한 한국어 번역을 제공하는 FastAPI 백엔드입니다.

- 저장소: https://github.com/Raising-Lee-Hyun-seong/K-aption_BE
- STT: faster-whisper
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

STT와 번역 결과를 터미널에 반환하는 CLI는 사용하지 않습니다. 프론트엔드는 HTTP API의 JSON 응답을 사용합니다.

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

- Swagger UI: http://localhost:8000/docs
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

응답에는 인식 언어와 확률, `start`, `end`, `text`를 가진 구간 목록, 처리 시간이 포함됩니다. 지원 확장자는 MP4, MOV, MKV, WEBM, WAV, MP3, M4A입니다. 업로드 파일은 임시 파일로 처리한 뒤 요청 종료 시 삭제합니다. STT 요청도 한 번에 하나씩 실행됩니다.

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
3. STT 결과를 문맥 단위로 묶어 번역 API에 연결합니다.
4. 전공 용어집 20개와 번역 품질 평가 데이터를 준비합니다.
5. SRT/VTT 생성과 영상 자막 재생 API를 추가합니다.

## 참고 자료

- [KoreanLM 원본 코드](https://github.com/quantumaikr/KoreanLM)
- [KoreanLM 모델 카드](https://huggingface.co/quantumaikr/KoreanLM)
- [KoreanLM 모델 설정](https://huggingface.co/quantumaikr/KoreanLM/blob/main/config.json)
- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)

KoreanLM 기반 코드의 라이선스는 저장소의 `LICENSE`를 확인하세요. 강의 영상과 생성 자막의 이용 조건은 코드와 별도로 관리합니다.
