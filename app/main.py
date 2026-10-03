"""FastAPI 서버의 모델 생명주기, CORS, 상태 조회 및 STT·한국어 번역 엔드포인트를 구성한다."""
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.schemas import (
    HealthResponse,
    TranscriptionResponse,
    TranslationRequest,
    TranslationResponse,
)
from app.services.transcription import TranscriptionService
from app.services.translation import TranslationService

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("KAPTION_MODEL_DIR", ROOT / "models/koreanlm-4bit"))
TEMPLATE_PATH = ROOT / "templates/korean.json"
ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".wav", ".mp3", ".m4a"}
API_DESCRIPTION = """
K-aption은 해외 대학 강의의 영어 음성을 인식하고 한국어로 번역합니다.

- **Transcription**: 영상·음성 파일을 faster-whisper로 인식합니다.
- **Translation**: 영어 문장과 주변 문맥을 KoreanLM 4bit 모델로 번역합니다.

추론 요청은 M4·16GB 환경의 메모리 사용을 고려해 서비스별로 직렬 처리합니다.
"""
OPENAPI_TAGS = [
    {
        "name": "System",
        "description": "서버와 모델 준비 상태를 확인합니다.",
    },
    {
        "name": "Translation",
        "description": "영어 강의 문장을 주변 문맥과 함께 한국어로 번역합니다.",
    },
    {
        "name": "Transcription",
        "description": "영상·음성 파일에서 영어 원문과 구간 시간을 추출합니다.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 번역·STT 서비스를 준비하고 종료 시 애플리케이션 참조를 해제한다."""
    app.state.translator = TranslationService(MODEL_DIR, TEMPLATE_PATH)
    app.state.transcriber = TranscriptionService(
        model_name=os.getenv("KAPTION_STT_MODEL", "small"),
        device=os.getenv("KAPTION_STT_DEVICE", "cpu"),
        compute_type=os.getenv("KAPTION_STT_COMPUTE_TYPE", "int8"),
    )
    yield
    app.state.translator = None
    app.state.transcriber = None


app = FastAPI(
    title="K-aption API",
    summary="해외 대학 강의용 영어 STT·한국어 번역 API",
    description=API_DESCRIPTION,
    version="0.1.0",
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=OPENAPI_TAGS,
    license_info={"name": "Apache 2.0", "identifier": "Apache-2.0"},
    lifespan=lifespan,
)

origins = [
    origin.strip()
    for origin in os.getenv(
        "KAPTION_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def openapi_schema() -> dict:
    """OpenAPI를 생성하고 자막 예시의 명시적 null 값을 보존해 Swagger에 제공한다."""
    if app.openapi_schema is None:
        schema = get_openapi(
            title=app.title,
            version=app.version,
            summary=app.summary,
            description=app.description,
            routes=app.routes,
            tags=app.openapi_tags,
            license_info=app.license_info,
        )
        schema["components"]["schemas"]["TranscriptionResponse"]["examples"] = (
            TranscriptionResponse.model_config["json_schema_extra"]["examples"]
        )
        app.openapi_schema = schema
    return app.openapi_schema


app.openapi = openapi_schema


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="서버 및 모델 상태 확인",
)
def health(request: Request) -> HealthResponse:
    """서버 상태와 번역·STT 모델의 로딩 여부를 반환한다."""
    translator = getattr(request.app.state, "translator", None)
    transcriber = getattr(request.app.state, "transcriber", None)
    return HealthResponse(
        status="ok" if translator is not None else "starting",
        translation_model_loaded=translator is not None,
        transcription_model_loaded=bool(transcriber and transcriber.loaded),
    )


@app.post(
    "/api/v1/translations",
    response_model=TranslationResponse,
    tags=["Translation"],
    summary="영어 강의 문장 번역",
    description="대상 영어 문장과 선택적 주변 문맥을 받아 한국어 번역을 반환합니다.",
    responses={422: {"description": "요청 형식, 모델 토큰 제한 또는 번역 생성 검증 오류"}},
)
async def translate(body: TranslationRequest, request: Request) -> TranslationResponse:
    """요청 문장을 문맥과 함께 번역하고 입력·생성 검증 오류를 HTTP 422로 반환한다."""
    try:
        result = await run_in_threadpool(
            request.app.state.translator.translate,
            body.text,
            body.context,
            body.max_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return TranslationResponse(**result)


@app.post(
    "/api/v1/transcriptions",
    response_model=TranscriptionResponse,
    tags=["Transcription"],
    summary="영상·음성 영어 STT",
    description=(
        "업로드한 미디어를 구간별 영어 원문과 타임스탬프로 변환하고, "
        "각 영어 원문을 한국어로 번역합니다. 번역 검증에 실패한 구간은 "
        "translation=null, translation_error=translation_failed로 반환하며 처리를 계속합니다. "
        "음성이 없으면 segments=[]를 반환합니다. 모든 시각의 단위는 초입니다."
    ),
    responses={
        415: {"description": "지원하지 않는 미디어 확장자", "content": {
            "application/json": {"example": {"detail": "Unsupported media file extension"}}
        }},
        422: {"description": "요청 형식 또는 오디오 디코딩·검증 오류", "content": {
            "application/json": {"examples": {
                "audio": {"summary": "오디오 디코딩 실패", "value": {
                    "detail": "The uploaded media does not contain decodable audio"}},
                "validation": {"summary": "필수 업로드 파일 누락", "value": {
                    "detail": [{"type": "missing", "loc": ["body", "file"],
                                "msg": "Field required", "input": None}]}},
            }}
        }},
        500: {"description": "예기치 않은 추론 또는 서버 오류. 전체 요청 실패"},
    },
)
async def transcribe(
    request: Request,
    file: UploadFile = File(description="인식할 영상 또는 음성 파일"),
    language: Literal["en"] = Query(
        default="en",
        description="인식 언어. 현재 영어(en)만 지원하며 생략 시 en",
    ),
) -> TranscriptionResponse:
    """STT 후 구간별 번역 검증 실패를 결과에 기록하고 업로드·임시 파일을 정리한다."""
    temp_path = None
    try:
        suffix = Path(file.filename or "upload").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=415, detail="Unsupported media file extension")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            await run_in_threadpool(shutil.copyfileobj, file.file, temp_file)
        try:
            result = await run_in_threadpool(
                request.app.state.transcriber.transcribe,
                temp_path,
                language,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        previous_translation = ""
        translated_segments = []
        for segment in result["segments"]:
            try:
                translated = await run_in_threadpool(
                    request.app.state.translator.translate,
                    segment["text"],
                    previous_translation,
                    128,
                )
            except ValueError:
                translated_segments.append({
                    **segment, "translation": None,
                    "translation_error": "translation_failed",
                })
                previous_translation = ""
            else:
                translated_segments.append({
                    **segment, "translation": translated["translation"],
                    "translation_error": None,
                })
                previous_translation = translated["translation"]
        return TranscriptionResponse(segments=translated_segments)
    finally:
        try:
            await file.close()
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
