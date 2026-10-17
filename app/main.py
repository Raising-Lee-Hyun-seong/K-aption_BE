"""모델·작업자 생명주기와 동기 STT·번역 및 비동기 작업 API를 구성한다."""
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from starlette.responses import StreamingResponse

from app.middleware import TimingMiddleware
from app.schemas import (
    CapabilitiesResponse,
    HealthResponse,
    JobStatusResponse,
    TranscriptionResponse,
    TranslationRequest,
    TranslationResponse,
    UploadCapabilities,
)
from app.services.jobs import (
    JobCapacityError, JobManager, JobNotReadyError, JobStorageError, UnsupportedMediaError,
)
from app.services.job_events import job_events
from app.services.pipeline import MediaValidationError, process_media
from app.services.subtitle_export import render_subtitles
from app.services.media import (
    MAX_MEDIA_SECONDS,
    MAX_UPLOAD_BYTES,
    ALLOWED_SUFFIXES,
    UPLOAD_TOO_LARGE,
    UploadTooLargeError,
    copy_upload,
)
from app.services.transcription import TranscriptionService
from app.services.translation import TranslationService
from timing import measure

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("KAPTION_MODEL_DIR", ROOT / "models/koreanlm-4bit"))
TEMPLATE_PATH = ROOT / "templates/korean.json"
API_DESCRIPTION = """
K-aption은 해외 대학 강의의 영어 음성을 인식하고 한국어로 번역합니다.

- **Transcription**: 영상·음성 파일을 faster-whisper로 인식합니다.
- **Translation**: 영어 문장과 주변 문맥을 KoreanLM으로 번역합니다. 기본 Mac은 MLX 4bit, CPU 컨테이너는 동일 가중치의 GGUF Q4_1을 사용합니다.

추론 요청은 로컬 장비의 메모리 사용을 고려해 서비스별로 직렬 처리합니다.
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
    {"name": "Jobs", "description": "작업을 접수하고 단계·번역 진행률·저장 결과를 조회합니다."},
]


def create_translation_service(model_path: Path, template_path: Path):
    """실행 환경에 맞는 추론 엔진을 선택하고 같은 번역 계약의 서비스를 만든다.

    :param model_path: mlx는 기존 4bit 디렉터리, llama-cpp는 재포장한 GGUF 파일 경로.
    :param template_path: 공통 프롬프트 JSON 경로.
    :return: 공통 검증·재시도를 제공하는 번역 서비스.
    :raises ValueError: 백엔드·장치·스레드 설정 또는 모델 구조가 잘못된 경우.
    :raises OSError: 모델이나 공통 설정 파일을 읽을 수 없는 경우.

    기본 mlx 동작을 유지한다. llama-cpp는 GPU 오프로딩 없이 CPU를 사용한다.
    """
    backend = os.getenv('KAPTION_TRANSLATION_BACKEND', 'mlx')
    if backend == 'mlx':
        return TranslationService(model_path, template_path,
                                  device=os.getenv('KAPTION_TRANSLATION_DEVICE', 'auto'))
    if backend == 'llama-cpp':
        from app.services.translation_llamacpp import LlamaCppTranslationService
        return LlamaCppTranslationService(model_path, template_path,
                                         threads=int(os.getenv('KAPTION_CPU_THREADS', '4')))
    raise ValueError('Translation backend must be mlx or llama-cpp')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """공유 모델과 단일 작업자를 시작하고 정상 종료 시 추론 종료·파일 정리를 기다린다.

    :param app: 서비스를 저장할 FastAPI 또는 호출할 하위 ASGI 애플리케이션.
    :return: 비동기 컨텍스트 관리자. 진입 시 서비스 준비가 끝난 시점의 None. 컨텍스트 종료 시 앱의 서비스 참조를 해제한다.
    :yield: 서비스 준비가 끝난 시점의 None. 컨텍스트 종료 시 앱의 서비스 참조를 해제한다.
    :raises RuntimeError: 작업 저장소를 다른 서버 작업자가 이미 사용 중일 때.
    :raises OSError: 작업 저장소 생성·복구·정리가 실패할 때.

    모델을 로드하고 작업 저장소를 생성·복구한다. 정상 종료는 현재 작업을 기다리고
    대기 작업을 실패 처리하며 입력을 삭제한다. 같은 저장소는 서버 worker 하나만 사용한다.
    """
    manager = JobManager(
        Path(os.getenv('KAPTION_JOB_DIR', ROOT / 'outputs/jobs')),
        None, None,
        max_active=int(os.getenv('KAPTION_MAX_ACTIVE_JOBS', '4')),
        ttl_seconds=int(os.getenv('KAPTION_JOB_TTL_SECONDS', '86400')),
    )
    app.state.jobs = manager
    try:
        await manager.start()
        app.state.translator = create_translation_service(MODEL_DIR, TEMPLATE_PATH)
        app.state.transcriber = TranscriptionService(
            model_name=os.getenv('KAPTION_STT_MODEL', 'small'),
            device=os.getenv('KAPTION_STT_DEVICE', 'cpu'),
            compute_type=os.getenv('KAPTION_STT_COMPUTE_TYPE', 'int8'),
        )
        manager.transcriber = app.state.transcriber
        manager.translator = app.state.translator
        yield
    finally:
        try:
            await manager.close()
        finally:
            try:
                close = getattr(getattr(app.state, 'translator', None), 'close', None)
                if callable(close):
                    close()
            finally:
                app.state.jobs = None
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
    expose_headers=['Location', 'Retry-After', 'Content-Disposition'],
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


@app.get('/api/v1/capabilities', response_model=CapabilitiesResponse, tags=['System'],
         summary='서버 업로드 입력 규칙 조회')
def get_capabilities(response: Response) -> CapabilitiesResponse:
    """실제 검증과 공유하는 확장자·크기·재생 시간 제한과 STT 언어를 반환한다.

    :param response: 배포 전 입력 규칙의 캐시 재사용을 막을 헤더를 설정하는 응답.
    :return: 동기·작업 업로드의 프론트엔드 사전 검사에 사용할 입력 규칙.

    모델·작업자를 조회하거나 로드하지 않고 파일 저장·미디어 디코딩도 하지 않는다.
    클라이언트 검사는 안내용이며 업로드 시 서버 검증을 그대로 수행한다.
    """
    response.headers['Cache-Control'] = 'no-store'
    return CapabilitiesResponse(upload=UploadCapabilities(
        allowed_extensions=sorted(ALLOWED_SUFFIXES),
        max_upload_bytes=MAX_UPLOAD_BYTES,
        max_duration_seconds=MAX_MEDIA_SECONDS,
        supported_languages=['en'],
    ))


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
        "파일은 1GiB 이하·재생 길이 90분 이하만 허용합니다. "
        "인접한 STT 조각을 문장 단위로 묶어 한국어로 번역합니다. "
        "묶음의 첫 시작·마지막 종료 시각을 반환하므로 원래 STT 구간 수와 다를 수 있습니다. "
        "번역 검증에 실패한 구간은 "
        "translation=null, translation_error=translation_failed로 반환하며 처리를 계속합니다. "
        "음성이 없으면 segments=[]를 반환합니다. 모든 시각의 단위는 초입니다."
    ),
    responses={
        413: {"description": "업로드 파일이 1GiB를 초과", "content": {
            "application/json": {"example": {"detail": UPLOAD_TOO_LARGE}}
        }},
        415: {"description": "지원하지 않는 미디어 확장자", "content": {
            "application/json": {"example": {"detail": "Unsupported media file extension"}}
        }},
        422: {"description": "요청 형식, 90분 초과 또는 오디오 디코딩·검증 오류", "content": {
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
    :param file: 1GiB 이하·90분 이하의 업로드 미디어. 처리가 끝나면 닫는다.
    :param language: 인식할 음성 언어 코드. 기본값은 영어 en이다.
    :return: 영어 원문·한국어 번역·초 단위 시각·구간 오류를 담은 TranscriptionResponse.
    :raises HTTPException: 크기 초과는 413, 미지원 확장자는 415, 길이 초과·오디오 디코딩·검증 실패는 422로 발생한다. 구간별 번역 검증 실패는 결과에 기록한다.

    업로드를 임시 파일에 저장하며 성공·실패 모두 업로드를 닫고 임시 파일을 삭제한다.
    """
    temp_path = None
    try:
        suffix = Path(file.filename or "upload").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=415, detail="Unsupported media file extension")
        if file.size is not None and file.size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=UPLOAD_TOO_LARGE)

        try:
            with measure("upload_save") as metrics:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                    metrics["upload_bytes"] = await run_in_threadpool(
                        copy_upload, file.file, temp_file,
                    )
        except UploadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        try:
            return await process_media(
                temp_path, request.app.state.transcriber, request.app.state.translator, language,
            )
        except MediaValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        try:
            await file.close()
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def job_manager(request: Request) -> JobManager:
    """서버에 준비된 작업 관리자를 반환한다.

    :param request: 작업 관리자가 저장된 FastAPI 요청.
    :return: 현재 앱의 작업 관리자.
    :raises HTTPException: 작업 관리자가 준비되지 않았으면 HTTP 503.
    """
    manager = getattr(request.app.state, 'jobs', None)
    if manager is None:
        raise HTTPException(status_code=503, detail='Job worker is unavailable')
    return manager


@app.post('/api/v1/jobs', status_code=202, response_model=JobStatusResponse, tags=['Jobs'],
          summary='영상·음성 처리 작업 접수', responses={
              413: {'description': '1GiB 크기 초과'},
              415: {'description': '미지원 확장자'},
              422: {'description': '요청·미디어 열기·알려진 90분 길이 검증 실패'},
              503: {'description': '작업자 미준비·종료 중 또는 미완료 작업 한도 초과'},
          })
async def create_job(
    request: Request, response: Response,
    file: UploadFile = File(description='영어 영상·음성. 1GiB·90분 이하'),
    language: Literal['en'] = Query(default='en'),
) -> JobStatusResponse:
    """업로드를 저장·검증한 뒤 연결과 독립적으로 처리할 작업을 접수한다.

    :param request: 작업 관리자가 저장된 요청.
    :param response: 상태 조회 경로를 Location 헤더에 추가할 HTTP 응답.
    :param file: 처리할 업로드 파일. 성공·실패 모두 닫는다.
    :param language: 인식 언어. en만 허용한다.
    :return: 작업 ID와 접수 상태. HTTP 202는 번역 완료를 뜻하지 않는다.
    :raises HTTPException: 크기 413·확장자 415·미디어 422·용량 또는 작업자 503 오류.

    작업 저장소에 미디어·상태를 저장한다. 전체 디코딩에서만 확인되는 오류는
    접수 후 작업의 failed 상태로 제공한다.
    """
    try:
        record = await job_manager(request).submit(file, language)
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UnsupportedMediaError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except MediaValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except JobCapacityError as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={'Retry-After': '5'}) from exc
    finally:
        await file.close()
    response.headers['Location'] = f'/api/v1/jobs/{record.job_id}'
    return record


@app.get('/api/v1/jobs/{job_id}', response_model=JobStatusResponse, tags=['Jobs'],
         summary='작업 상태·번역 진행률 조회', responses={404: {'description': '없는 작업 또는 만료'}})
async def get_job(job_id: str, request: Request) -> JobStatusResponse:
    """작업의 실제 단계·구간 진행률·공개 실패 사유를 반환한다.

    :param job_id: 접수 시 받은 작업 ID.
    :param request: 작업 관리자가 저장된 요청.
    :return: 현재 작업 상태와 대기 포함 경과 시간.
    :raises HTTPException: 작업이 없거나 만료됐으면 404, 관리자 미준비는 503.
    """
    try:
        return job_manager(request).get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='Job not found') from exc


@app.post('/api/v1/jobs/{job_id}/cancel', response_model=JobStatusResponse, tags=['Jobs'],
          summary='대기·실행 작업 취소 요청', responses={
              200: {'description': '취소 완료 또는 이미 취소된 작업'},
              202: {'description': '현재 추론 단계가 끝나면 취소'},
              404: {'description': '없는 작업 또는 만료'},
              409: {'description': '완료·실패 작업 또는 아직 업로드 접수 중'},
              500: {'description': '취소 요청·파일·상태 저장 실패'},
              503: {'description': '작업 관리자 미준비'},
          })
async def cancel_job(job_id: str, request: Request, response: Response) -> JobStatusResponse:
    """대기 작업을 취소하거나 실행 중 작업에 안전한 단계 경계 취소를 요청한다.

    :param job_id: 접수 후 받은 취소 대상 작업 ID.
    :param request: 작업 관리자가 저장된 요청.
    :param response: 실행 중 취소 요청의 202 상태 코드와 캐시 헤더를 설정할 응답.
    :return: 취소 완료는 200·cancelled, 실행 중은 202·cancelling 상태.
    :raises HTTPException: 없음·만료 404, 종료 상태 409, 저장 오류 500, 관리자 미준비 503.

    취소 요청을 저장하며 입력 삭제는 추론이 사용하지 않는 시점에 수행한다.
    cancelled·cancelling 작업의 재요청은 해당 상태를 그대로 반환한다.
    """
    try:
        record = job_manager(request).cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='Job not found') from exc
    except JobNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (JobStorageError, OSError) as exc:
        raise HTTPException(status_code=500, detail='Job cancellation could not be saved') from exc
    response.status_code = 202 if record.status == 'cancelling' else 200
    response.headers['Cache-Control'] = 'no-store'
    return record


@app.get('/api/v1/jobs/{job_id}/events', response_class=StreamingResponse, tags=['Jobs'],
         summary='작업 상태·진행률 SSE 구독', responses={
             200: {'description': '상태 스냅샷과 heartbeat', 'content': {'text/event-stream': {}}},
             404: {'description': '없는 작업 또는 만료'},
             503: {'description': '작업 관리자 미준비'},
         })
async def get_job_events(job_id: str, request: Request) -> StreamingResponse:
    """현재 상태를 먼저 확인한 뒤 완료·실패·취소 완료까지 변경을 SSE로 제공한다.

    :param job_id: 구독할 기존 작업 ID.
    :param request: 작업 관리자가 저장된 요청.
    :return: status 이벤트와 heartbeat를 제공하는 text/event-stream 응답.
    :raises HTTPException: 최초 조회에서 없음·만료는 404, 관리자 미준비는 503.

    연결 종료는 구독만 끝내며 작업을 취소하지 않는다. 재연결 시 현재 스냅샷부터
    시작하고 이력은 재생하지 않는다. 완료·실패·취소 완료는 한 번 보내고 연결을 닫는다.
    """
    manager = job_manager(request)
    try:
        initial = manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='Job not found') from exc
    return StreamingResponse(job_events(manager, initial), media_type='text/event-stream', headers={
        'Cache-Control': 'no-cache, no-store',
        'X-Accel-Buffering': 'no',
    })


@app.get('/api/v1/jobs/{job_id}/result', response_model=TranscriptionResponse, tags=['Jobs'],
         summary='완료된 작업 자막 결과 조회', responses={
             404: {'description': '없는 작업 또는 만료'},
             409: {'description': '대기·실행·실패·취소 상태라 완료 결과 없음'},
             500: {'description': '저장된 결과 없음 또는 손상'},
         })
async def get_job_result(job_id: str, request: Request) -> TranscriptionResponse:
    """완료 작업의 저장된 자막을 기존 동기 API와 같은 형식으로 반환한다.

    :param job_id: 접수 시 받은 작업 ID.
    :param request: 작업 관리자가 저장된 요청.
    :return: 성공·부분 실패·무음을 구분할 수 있는 자막 구간 응답.
    :raises HTTPException: 없음·만료 404, 미완료·실패·취소 409, 저장 손상 500, 관리자 미준비 503.
    """
    try:
        return job_manager(request).result(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='Job not found') from exc
    except JobNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except JobStorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get('/api/v1/jobs/{job_id}/subtitles', response_class=Response, tags=['Jobs'],
         summary='완료 작업의 SRT·VTT 자막 다운로드', responses={
             200: {'description': 'UTF-8 자막 문서', 'content': {
                 'application/x-subrip': {}, 'text/vtt': {}}},
             404: {'description': '없는 작업 또는 만료'},
             409: {'description': '작업 미완료·실패·취소'},
             422: {'description': '지원하지 않는 형식·언어·다운로드 옵션'},
             500: {'description': '저장 결과 손상 또는 유효하지 않은 자막'},
             503: {'description': '작업 관리자 미준비'},
         })
async def get_job_subtitles(
    job_id: str, request: Request,
    file_format: Literal['srt', 'vtt'] = Query(default='srt', alias='format'),
    language: Literal['ko', 'en', 'both'] = Query(default='ko'),
    download: bool = Query(default=True, description='false이면 브라우저 트랙용 inline 응답'),
) -> Response:
    """저장된 결과를 자막 문서로 변환하고 다운로드 또는 트랙용 응답을 반환한다.

    :param job_id: 완료된 작업 ID.
    :param request: 작업 관리자가 저장된 요청.
    :param file_format: format 쿼리의 srt 또는 vtt. 기본 srt.
    :param language: ko·en·both 중 하나. 기본 ko이며 번역 실패 구간은 원문을 제공한다.
    :param download: 참이면 attachment, 거짓이면 inline Content-Disposition을 사용한다.
    :return: UTF-8 자막 본문·파일명·미디어 유형이 포함된 HTTP 응답.
    :raises HTTPException: 없음·만료 404, 미완료·실패·취소 409, 옵션 422, 손상 500, 관리자 미준비 503.

    기존 결과를 읽기만 하고 모델 재추론이나 자막 파일 저장은 하지 않는다.
    """
    result = await get_job_result(job_id, request)
    try:
        content = render_subtitles(result, file_format, language)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail='Stored subtitles cannot be exported') from exc
    filename = f'kaption-{job_id}-{language}.{file_format}'
    disposition = 'attachment' if download else 'inline'
    media_type = 'text/vtt' if file_format == 'vtt' else 'application/x-subrip; charset=utf-8'
    return Response(content=content, media_type=media_type, headers={
        'Content-Disposition': f'{disposition}; filename="{filename}"',
        'Cache-Control': 'no-store',
    })
