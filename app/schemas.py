from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    translation_model_loaded: bool
    transcription_model_loaded: bool


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8_000)
    context: str = Field(default="", max_length=16_000)
    max_tokens: int = Field(default=128, ge=1, le=1_024)


class TranslationResponse(BaseModel):
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
