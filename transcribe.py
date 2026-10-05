"""미디어 오디오를 검증하고 faster-whisper로 음성 구간과 타임스탬프를 추출한다."""
from pathlib import Path
from threading import Lock
from time import perf_counter

from timing import measure

class TranscriptionService:
    """Whisper 모델을 지연 로드하고 직렬 추론으로 타임스탬프가 있는 음성 구간을 제공한다."""
    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """모델·장치·연산 형식을 저장하고 모델 로딩 및 추론 잠금을 준비한다.

        :param model_name: 사용할 Whisper 모델 이름 또는 로컬 모델 경로.
        :param device: Whisper 추론 장치. 예: cpu, cuda.
        :param compute_type: Whisper 연산 정밀도. 예: int8.
        :return: None.
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def loaded(self) -> bool:
        """Whisper 모델이 메모리에 로드되었는지 반환한다.

        :return: Whisper 모델을 이미 로드했으면 True, 아니면 False.
        """
        return self._model is not None

    def _get_model(self):
        """동시 로딩을 방지하며 Whisper 모델을 최초 한 번 생성해 재사용한다.

        :return: 한 번 로드해 재사용하는 faster_whisper.WhisperModel 인스턴스.
        """
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    with measure("stt_model_load", model=self.model_name):
                        from faster_whisper import WhisperModel

                        self._model = WhisperModel(
                            self.model_name,
                            device=self.device,
                            compute_type=self.compute_type,
                        )
        return self._model

    def transcribe(self, media_path: Path, language: str = "en") -> dict:
        """오디오를 디코딩·검증하고 음성 구간을 실제 길이 안의 타임스탬프와 함께 반환한다.

        :param media_path: 오디오를 추출할 영상 또는 음성 파일 경로.
        :param language: 인식할 음성 언어 코드. 기본값은 영어 en이다.
        :return: language·language_probability·segments(start/end는 초, text는 원문)·elapsed_seconds(추론 대기 포함 초) 사전.
        :raises ValueError: 오디오 디코딩 실패·빈 오디오·비유한 샘플일 때.
        """
        import numpy as np
        from av.error import FFmpegError
        from faster_whisper.audio import decode_audio

        with measure("stt_decode"):
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
        with measure("stt_queue"):
            self._inference_lock.acquire()
        try:
            with measure("stt_inference", audio_seconds=duration) as metrics:
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
                metrics["segment_count"] = len(segments)
        finally:
            self._inference_lock.release()
        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": segments,
            "elapsed_seconds": perf_counter() - start,
        }
