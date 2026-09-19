import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

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


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    version="0.1.0",
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


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    translator = getattr(request.app.state, "translator", None)
    transcriber = getattr(request.app.state, "transcriber", None)
    return HealthResponse(
        status="ok" if translator is not None else "starting",
        translation_model_loaded=translator is not None,
        transcription_model_loaded=bool(transcriber and transcriber.loaded),
    )


@app.post("/api/v1/translations", response_model=TranslationResponse)
async def translate(body: TranslationRequest, request: Request) -> TranslationResponse:
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


@app.post("/api/v1/transcriptions", response_model=TranscriptionResponse)
async def transcribe(
    request: Request,
    file: UploadFile,
    language: str = "en",
) -> TranscriptionResponse:
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Unsupported media file extension")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            await run_in_threadpool(shutil.copyfileobj, file.file, temp_file)
        result = await run_in_threadpool(
            request.app.state.transcriber.transcribe,
            temp_path,
            language,
        )
        return TranscriptionResponse(**result)
    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
