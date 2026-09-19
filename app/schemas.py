from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    translation_model_loaded: bool
    transcription_model_loaded: bool


class TranslationRequest(BaseModel):
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
    context: str = Field(default="", max_length=16_000)
    max_tokens: int = Field(default=128, ge=1, le=1_024)


class TranslationResponse(BaseModel):
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
    generated_tokens: int
    elapsed_seconds: float


class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    language: str
    language_probability: float
    segments: list[TranscriptionSegment]
    elapsed_seconds: float
