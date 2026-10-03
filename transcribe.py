"""미디어 오디오를 검증하고 faster-whisper로 음성 구간과 타임스탬프를 추출한다."""
from pathlib import Path
from threading import Lock
from time import perf_counter


class TranscriptionService:
    """Whisper 모델을 지연 로드하고 직렬 추론으로 타임스탬프가 있는 음성 구간을 제공한다."""
    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """모델·장치·연산 형식을 저장하고 모델 로딩 및 추론 잠금을 준비한다."""
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def loaded(self) -> bool:
        """Whisper 모델이 메모리에 로드되었는지 반환한다."""
        return self._model is not None

    def _get_model(self):
        """동시 로딩을 방지하며 Whisper 모델을 최초 한 번 생성해 재사용한다."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from faster_whisper import WhisperModel

                    self._model = WhisperModel(
                        self.model_name,
                        device=self.device,
                        compute_type=self.compute_type,
                    )
        return self._model

    def transcribe(self, media_path: Path, language: str = "en") -> dict:
        """오디오를 디코딩·검증하고 음성 구간을 실제 길이 안의 타임스탬프와 함께 반환한다."""
        import numpy as np
        from av.error import FFmpegError
        from faster_whisper.audio import decode_audio

        try:
            audio = decode_audio(str(media_path))
        except (FFmpegError, OSError) as exc:
            raise ValueError("The uploaded media does not contain decodable audio") from exc
        if audio.size == 0:
            raise ValueError("The uploaded media does not contain decodable audio")
        if not np.isfinite(audio).all():
            raise ValueError("The uploaded media contains invalid audio samples")
        duration = audio.size / 16_000

        model = self._get_model()
        start = perf_counter()
        with self._inference_lock:
            raw_segments, info = model.transcribe(
                audio,
                language=language,
                task="transcribe",
                beam_size=5,
                vad_filter=True,
            )
            segments = [
                {
                    "start": max(0.0, segment.start),
                    "end": min(duration, segment.end),
                    "text": segment.text.strip(),
                }
                for segment in raw_segments
                if segment.text.strip()
                and min(duration, segment.end) > max(0.0, segment.start)
            ]
        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": segments,
            "elapsed_seconds": perf_counter() - start,
        }
