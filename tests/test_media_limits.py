"""파일 크기·재생 길이 경계와 제한 초과 시 추론 생략 및 업로드 정리를 검증한다."""
import io
import tempfile
import unittest
import wave
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi import HTTPException, UploadFile
from starlette.requests import Request

from app.main import transcribe
from app.services.media import (
    MEDIA_TOO_LONG,
    UploadTooLargeError,
    copy_upload,
    validate_media_duration,
)
from transcribe import TranscriptionService


class MediaLimitTests(unittest.TestCase):
    """크기 상한·메타데이터·실제 오디오 길이를 모델 실행 없이 검증한다."""

    def test_copy_accepts_boundary_and_stops_before_extra_byte(self):
        """크기 상한은 허용하고 추가 바이트가 있으면 상한 이내 데이터만 저장한다.

        :return: None.
        :raises AssertionError: 복사 경계 또는 초과 파일 저장 중단이 기대와 다를 때.
        """
        with patch('app.services.media.MAX_UPLOAD_BYTES', 4):
            output = io.BytesIO()
            self.assertEqual(copy_upload(io.BytesIO(b'1234'), output), 4)
            self.assertEqual(output.getvalue(), b'1234')
            output = io.BytesIO()
            with self.assertRaises(UploadTooLargeError):
                copy_upload(io.BytesIO(b'12345'), output)
            self.assertLessEqual(len(output.getvalue()), 4)

    def test_duration_checks_container_and_audio_video_streams(self):
        """90분 경계·컨테이너·스트림 초과와 길이 메타데이터 누락을 구분한다.

        :return: None.
        :raises AssertionError: 알려진 길이 초과를 허용하거나 허용 경계를 거부할 때.
        """
        for seconds, stream_type, rejected in (
            (5400, None, False), (5401, None, True),
            (5401, 'audio', True), (5401, 'video', True),
            (None, None, False),
        ):
            with self.subTest(seconds=seconds, stream_type=stream_type):
                container = SimpleNamespace(
                    duration=seconds * 1_000_000 if seconds is not None and stream_type is None else None,
                    streams=[] if stream_type is None else [SimpleNamespace(
                        type=stream_type, duration=seconds * 1000, time_base=Fraction(1, 1000))],
                )
                with patch('av.open') as opened:
                    opened.return_value.__enter__.return_value = container
                    if rejected:
                        with self.assertRaisesRegex(ValueError, MEDIA_TOO_LONG):
                            validate_media_duration(Path('lecture.mp4'))
                    else:
                        validate_media_duration(Path('lecture.mp4'))
                    opened.return_value.__exit__.assert_called_once()

    def test_real_wav_duration_probe(self):
        """실제 WAV 파일을 열어 길이를 검사하고 낮춘 제한에서 초과를 감지한다.

        :return: None.
        :raises AssertionError: 실제 디코더의 길이 검사 또는 파일 종료가 실패할 때.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'clip.wav'
            with wave.open(str(path), 'wb') as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b'\x00\x00' * 16000)
            validate_media_duration(path)
            with patch('app.services.media.MAX_MEDIA_SECONDS', 0.5):
                with self.assertRaisesRegex(ValueError, MEDIA_TOO_LONG):
                    validate_media_duration(path)

    def test_metadata_rejection_skips_decode_and_model(self):
        """메타데이터 길이 초과 시 오디오 디코딩과 모델 로딩을 모두 생략한다.

        :return: None.
        :raises AssertionError: 초과 파일에 디코더 또는 모델을 호출할 때.
        """
        service = TranscriptionService()
        with patch('transcribe.validate_media_duration', side_effect=ValueError(MEDIA_TOO_LONG)), patch(
                'faster_whisper.audio.decode_audio') as decode, patch.object(service, '_get_model') as model:
            with self.assertRaisesRegex(ValueError, MEDIA_TOO_LONG):
                service.transcribe(Path('long.mp4'))
            decode.assert_not_called()
            model.assert_not_called()

    def test_decoded_duration_boundary_and_untrusted_metadata(self):
        """길이 메타데이터가 없어도 실제 샘플 수로 상한과 한 샘플 초과를 판정한다.

        :return: None.
        :raises AssertionError: 초과 오디오를 추론하거나 정확한 상한을 거부할 때.
        """
        for samples in (16000, 16001):
            service = TranscriptionService()
            model = Mock()
            model.transcribe.return_value = (iter([]), SimpleNamespace(
                language='en', language_probability=1.0))
            with patch('transcribe.validate_media_duration'), patch('transcribe.MAX_MEDIA_SECONDS', 1), patch(
                    'faster_whisper.audio.decode_audio', return_value=np.zeros(samples)), patch.object(
                    service, '_get_model', return_value=model) as load:
                if samples == 16000:
                    self.assertEqual(service.transcribe(Path('clip.wav'))['segments'], [])
                    load.assert_called_once()
                else:
                    with self.assertRaisesRegex(ValueError, MEDIA_TOO_LONG):
                        service.transcribe(Path('clip.wav'))
                    load.assert_not_called()


class UploadCleanupTests(unittest.IsolatedAsyncioTestCase):
    """크기 초과를 사전 또는 복사 중 발견해도 파일과 추론 자원을 정리한다."""

    async def test_size_rejection_closes_upload_and_removes_partial_file(self):
        """알려진 크기와 누락·축소된 크기 정보 모두 실제 상한을 적용하고 파일을 정리한다.

        :return: None.
        :raises AssertionError: 초과 업로드에서 추론이 실행되거나 임시 파일이 남을 때.
        """
        for size in (5, None, 1):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                upload = UploadFile(filename='lecture.wav', file=io.BytesIO(b'12345'), size=size)
                stt = SimpleNamespace(transcribe=Mock())
                translator = SimpleNamespace(translate=Mock())
                request = Request({'type': 'http', 'app': SimpleNamespace(
                    state=SimpleNamespace(transcriber=stt, translator=translator))})
                with patch('app.main.MAX_UPLOAD_BYTES', 4), patch('app.services.media.MAX_UPLOAD_BYTES', 4), patch(
                        'app.main.tempfile.tempdir', directory):
                    with self.assertRaises(HTTPException) as error:
                        await transcribe(request, upload, 'en')
                self.assertEqual(error.exception.status_code, 413)
                self.assertTrue(upload.file.closed)
                self.assertEqual(list(Path(directory).iterdir()), [])
                stt.transcribe.assert_not_called()
                translator.translate.assert_not_called()
