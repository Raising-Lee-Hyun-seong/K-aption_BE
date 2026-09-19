from pathlib import Path
from threading import Lock
from time import perf_counter


class TranscriptionService:
    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _get_model(self):
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
        import numpy as np
        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(media_path))
        if audio.size == 0:
            raise ValueError("The uploaded media does not contain decodable audio")
        if not np.isfinite(audio).all():
            raise ValueError("The uploaded media contains invalid audio samples")

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
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
                for segment in raw_segments
                if segment.text.strip()
            ]
        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": segments,
            "elapsed_seconds": perf_counter() - start,
        }
