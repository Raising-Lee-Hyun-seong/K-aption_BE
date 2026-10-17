"""상태 조회, 번역 및 자막 API의 요청·응답 형식과 입력 검증 규칙을 정의한다."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """서버 상태와 번역·STT 모델 로딩 여부를 표현한다."""
    status: str
    translation_model_loaded: bool
    transcription_model_loaded: bool


class TranslationRequest(BaseModel):
    """번역할 문장, 문맥 및 최대 생성 토큰 수의 입력 제약을 정의한다."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "The velocity is constant.",
                    "context": "We are discussing uniform motion.",
                    "max_tokens": 128,
                }
            ]
        }
    )

    text: str = Field(min_length=1, max_length=8_000)
    context: str = Field(default="", max_length=16_000,
                         description="앞 영어 원문 문맥. 마지막 256자만 사용")
    max_tokens: int = Field(default=128, ge=1, le=1_024)


class TranslationResponse(BaseModel):
    """번역문과 토큰 수·처리 시간의 응답 형식을 정의한다."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "translation": "속도는 일정합니다.",
                    "prompt_tokens": 240,
                    "generated_tokens": 13,
                    "elapsed_seconds": 1.2,
                }
            ]
        }
    )

    translation: str
    prompt_tokens: int
    generated_tokens: int = Field(description="성공한 마지막 시도에서 MLX가 실제 생성한 토큰 수")
    elapsed_seconds: float


class TranscriptionSegment(BaseModel):
    """문장으로 묶은 자막의 시각·원문과 번역 결과 또는 실패 코드를 표현한다."""
    start: float = Field(ge=0, description="영상 시작 기준 구간 시작 시각(초)")
    end: float = Field(gt=0, description="영상 시작 기준 구간 종료 시각(초)")
    text: str = Field(min_length=1, description="인접한 STT 조각을 문장 단위로 묶은 영어 원문")
    translation: Optional[str] = Field(description="한국어 번역문. 번역 검증 실패 시 null")
    translation_error: Optional[Literal["translation_failed"]] = Field(
        description="성공 시 null, 번역 검증 실패 시 translation_failed"
    )


class TranscriptionResponse(BaseModel):
    """성공·부분 실패·음성 없음 결과를 자막 구간 목록으로 표현한다."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"segments": [{"start": 3.6, "end": 7.44,
                               "text": "The velocity is constant.",
                               "translation": "속도는 일정합니다.",
                               "translation_error": None}]},
                {"segments": [{"start": 3.6, "end": 7.44,
                               "text": "The velocity is constant.",
                               "translation": None,
                               "translation_error": "translation_failed"}]},
                {"segments": []},
            ]
        }
    )
    segments: list[TranscriptionSegment] = Field(
        description="인식 순서의 자막 구간. 음성이 없으면 빈 배열"
    )


class JobError(BaseModel):
    """작업 실패의 공개 코드와 안전한 안내 메시지를 제공한다."""
    code: Literal['invalid_media', 'processing_failed', 'interrupted', 'storage_error']
    message: str


class JobStatusResponse(BaseModel):
    """작업 상태·취소 진행·측정 가능한 번역 진행률·결과 보관 기한을 표현한다."""
    job_id: str
    status: Literal['queued', 'running', 'cancelling', 'cancelled', 'completed', 'failed']
    stage: Literal['queued', 'transcribing', 'grouping', 'translating', 'cancelled', 'completed', 'failed']
    progress_percent: Optional[float] = Field(default=None, ge=0, le=100,
        description='번역 구간 처리 완료 비율. STT·대기 중에는 null, 작업 완료 시 100')
    completed_segments: int = Field(default=0, ge=0)
    total_segments: Optional[int] = Field(default=None, ge=0)
    failed_segments: int = Field(default=0, ge=0)
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = Field(default=None,
        description='완료·실패·취소 완료 시점부터의 결과 보관 기한. 미완료 작업은 null')
    elapsed_seconds: float = Field(default=0, ge=0,
        description='접수 시점부터 경과한 초. 대기 포함, 종료된 작업은 종료 시점까지')
    error: Optional[JobError] = None
