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

from app.middleware import TimingMiddleware
from app.schemas import (
    HealthResponse,
    TranscriptionResponse,
    TranslationRequest,
    TranslationResponse,
)
from app.services.subtitles import group_segments
from app.services.transcription import TranscriptionService
from app.services.translation import TranslationService
from timing import measure, segment_index

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
    """서버 시작 시 번역·STT 서비스를 준비하고 종료 시 애플리케이션 참조를 해제한다.

    :param app: 서비스를 저장할 FastAPI 또는 호출할 하위 ASGI 애플리케이션.
    :return: 비동기 컨텍스트 관리자. 진입 시 서비스 준비가 끝난 시점의 None. 컨텍스트 종료 시 앱의 서비스 참조를 해제한다.
    :yield: 서비스 준비가 끝난 시점의 None. 컨텍스트 종료 시 앱의 서비스 참조를 해제한다.
    """
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
app.add_middleware(TimingMiddleware)


def openapi_schema() -> dict:
    """OpenAPI를 생성하고 자막 예시의 명시적 null 값을 보존해 Swagger에 제공한다.

    :return: 자막 예시의 null을 보존한 캐시된 OpenAPI 스키마 사전.
    """
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
    """서버 상태와 번역·STT 모델의 로딩 여부를 반환한다.

    :param request: 서비스 인스턴스가 저장된 앱에 접근할 FastAPI 요청.
    :return: 서버 상태와 번역·STT 로딩 여부를 담은 HealthResponse.
    """
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
    """요청 문장을 문맥과 함께 번역하고 입력·생성 검증 오류를 HTTP 422로 반환한다.

    :param body: 영어 원문·문맥·최대 출력 토큰 수를 담은 TranslationRequest.
    :param request: 서비스 인스턴스가 저장된 앱에 접근할 FastAPI 요청.
    :return: 한국어 번역과 토큰 수·수행 시간을 담은 TranslationResponse.
    :raises HTTPException: 입력 토큰 제한 또는 생성 검증 실패 시 상태 코드 422로 발생한다.
    """
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
        "업로드한 미디어를 영어 원문과 타임스탬프로 변환하고, "
        "인접한 STT 조각을 문장 단위로 묶어 한국어로 번역합니다. "
        "묶음의 첫 시작·마지막 종료 시각을 반환하므로 원래 STT 구간 수와 다를 수 있습니다. "
        "번역 검증에 실패한 구간은 "
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
    """STT 조각을 문장으로 묶어 영어 문맥과 함께 번역하고 업로드 파일을 정리한다.

    문장별 번역 검증 실패는 원문·시각과 함께 결과에 남기며 이후 처리를 계속한다.
    문맥에는 이전 영어 원문만 사용해 실패하거나 잘못된 번역의 전파를 막는다.

    :param request: 서비스 인스턴스가 저장된 앱에 접근할 FastAPI 요청.
    :param file: 음성 인식할 업로드 미디어. 처리가 끝나면 닫는다.
    :param language: 인식할 음성 언어 코드. 기본값은 영어 en이다.
    :return: 영어 원문·한국어 번역·초 단위 시각·구간 오류를 담은 TranscriptionResponse.
    :raises HTTPException: 미지원 확장자는 415, 오디오 디코딩·검증 실패는 422로 발생한다. 구간별 번역 검증 실패는 결과에 기록한다.

    업로드를 임시 파일에 저장하며 성공·실패 모두 업로드를 닫고 임시 파일을 삭제한다.
    """
    temp_path = None
    try:
        suffix = Path(file.filename or "upload").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=415, detail="Unsupported media file extension")

        with measure("upload_save"):
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                await run_in_threadpool(shutil.copyfileobj, file.file, temp_file)
        try:
            with measure("stt") as metrics:
                result = await run_in_threadpool(
                    request.app.state.transcriber.transcribe,
                    temp_path,
                    language,
                )
                metrics["segment_count"] = len(result["segments"])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with measure("sentence_grouping", source_segment_count=len(result["segments"])) as metrics:
            sentences = group_segments(result["segments"])
            metrics["sentence_count"] = len(sentences)
        previous_source = ""
        translated_segments = []
        with measure(
            "translation_batch", source_segment_count=len(result["segments"]),
            sentence_count=len(sentences),
        ) as metrics:
            failed_segments = 0
            metrics["failed_segments"] = failed_segments
            for index, sentence in enumerate(sentences):
                segment = {key: sentence[key] for key in ("start", "end", "text")}
                token = segment_index.set(index)
                try:
                    with measure("segment_translation"):
                        translated = await run_in_threadpool(
                            request.app.state.translator.translate,
                            segment["text"],
                            previous_source,
                            128,
                        )
                except ValueError:
                    translated_segments.append({
                        **segment, "translation": None,
                        "translation_error": "translation_failed",
                    })
                    failed_segments += 1
                else:
                    translated_segments.append({
                        **segment, "translation": translated["translation"],
                        "translation_error": None,
                    })
                finally:
                    previous_source = segment["text"][-256:]
                    segment_index.reset(token)
                    metrics["failed_segments"] = failed_segments
        return TranscriptionResponse(segments=translated_segments)
    finally:
        try:
            await file.close()
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
