"""STT·번역 통합 흐름, 출력 재시도, 오디오 검증 및 업로드 자원 정리를 검증한다."""
import io
import sys
import tempfile
import unittest
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi import HTTPException, UploadFile
from starlette.requests import Request

from app.main import app, transcribe
from app.services.translation import TranslationService
from transcribe import TranscriptionService


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    """비동기 STT·번역 엔드포인트의 응답과 자원 정리를 검증한다."""
    async def run_upload(self, transcriber, translator, filename='lecture.wav'):
        """테스트 업로드를 처리하고 파일 종료와 임시 파일 삭제를 확인한다."""
        upload = UploadFile(filename=filename, file=io.BytesIO(b'audio'))
        request = Request({'type': 'http', 'app': SimpleNamespace(state=SimpleNamespace(
            transcriber=transcriber, translator=translator))})
        with tempfile.TemporaryDirectory() as directory:
            with patch('app.main.tempfile.tempdir', directory):
                try:
                    return await transcribe(request, upload, 'en')
                finally:
                    self.assertTrue(upload.file.closed)
                    self.assertEqual(list(Path(directory).iterdir()), [])

    async def test_segments_preserve_timestamps_and_previous_translation(self):
        """구간 순서·타임스탬프 보존과 직전 번역의 문맥 전달을 검증한다."""
        segments = [{'start': 1.0, 'end': 2.0, 'text': 'First.'},
                    {'start': 2.0, 'end': 3.5, 'text': 'Second.'}]
        stt = SimpleNamespace(transcribe=Mock(return_value={'segments': segments}))
        translator = SimpleNamespace(translate=Mock(side_effect=[
            {'translation': '첫 문장.'}, {'translation': '다음 문장.'}]))
        result = await self.run_upload(stt, translator)
        self.assertEqual(result.model_dump(), {'segments': [
            {**segments[0], 'translation': '첫 문장.', 'translation_error': None},
            {**segments[1], 'translation': '다음 문장.', 'translation_error': None}]})
        self.assertEqual(translator.translate.call_args_list[0].args, ('First.', '', 128))
        self.assertEqual(translator.translate.call_args_list[1].args, ('Second.', '첫 문장.', 128))

    async def test_errors_cleanup_upload_and_temp_file(self):
        """STT 검증 오류와 예기치 않은 번역 오류에서도 파일이 정리되는지 확인한다."""
        for stage in ('stt', 'unexpected'):
            with self.subTest(stage=stage):
                stt = SimpleNamespace(transcribe=Mock(return_value={'segments': [
                    {'start': 0, 'end': 1, 'text': 'Hello.'}]}))
                translator = SimpleNamespace(translate=Mock())
                if stage == 'stt':
                    stt.transcribe.side_effect = ValueError('bad audio')
                else:
                    translator.translate.side_effect = RuntimeError('inference error')
                with self.assertRaises(HTTPException if stage != 'unexpected' else RuntimeError) as error:
                    await self.run_upload(stt, translator)
                if stage != 'unexpected':
                    self.assertEqual(error.exception.status_code, 422)

    async def test_partial_failure_preserves_original_and_resets_context(self):
        """번역 실패 구간의 원문을 보존하고 문맥을 초기화한 뒤 다음 구간을 계속 처리한다."""
        segments = [{'start': i, 'end': i + 1, 'text': f'Sentence {i}.'}
                    for i in range(3)]
        translator = SimpleNamespace(translate=Mock(side_effect=[
            {'translation': '첫 문장.'}, ValueError('invalid output'),
            {'translation': '세 번째 문장.'}]))
        result = await self.run_upload(SimpleNamespace(transcribe=Mock(
            return_value={'segments': segments})), translator)
        self.assertEqual(result.segments[1].model_dump(), {
            **segments[1], 'translation': None, 'translation_error': 'translation_failed'})
        self.assertEqual(result.segments[2].translation, '세 번째 문장.')
        self.assertEqual(translator.translate.call_args_list[2].args[1], '')

    async def test_all_translations_fail_without_losing_stt(self):
        """모든 번역 검증이 실패해도 원문과 타임스탬프를 반환한다."""
        segment = {'start': 0, 'end': 1, 'text': 'Hello.'}
        result = await self.run_upload(SimpleNamespace(transcribe=Mock(
            return_value={'segments': [segment]})), SimpleNamespace(
                translate=Mock(side_effect=ValueError('invalid output'))))
        self.assertEqual(result.model_dump(), {'segments': [{
            **segment, 'translation': None, 'translation_error': 'translation_failed'}]})

    async def test_unsupported_extension(self):
        """지원하지 않는 확장자를 추론 없이 HTTP 415로 거부하는지 확인한다."""
        stt = SimpleNamespace(transcribe=Mock())
        with self.assertRaises(HTTPException) as error:
            await self.run_upload(stt, None, 'lecture.txt')
        self.assertEqual(error.exception.status_code, 415)
        stt.transcribe.assert_not_called()

    async def test_no_speech(self):
        """음성 구간이 없을 때 번역 호출 없이 빈 자막 목록을 반환하는지 확인한다."""
        translator = SimpleNamespace(translate=Mock())
        result = await self.run_upload(SimpleNamespace(transcribe=Mock(
            return_value={'segments': []})), translator)
        self.assertEqual(result.segments, [])
        translator.translate.assert_not_called()


class ServiceTests(unittest.TestCase):
    """번역 재시도·토큰 제한과 실제 오디오 디코딩 검증을 수행한다."""
    def translator(self):
        """실제 모델 로딩 없이 번역 로직을 검증할 테스트용 서비스를 구성한다."""
        service = TranslationService.__new__(TranslationService)
        service.model = object()
        service.tokenizer = SimpleNamespace(encode=lambda text: list(text))
        service.template = {'prompt_input': '{instruction}\n{input}'}
        service.sampler = object()
        service._lock = Lock()
        return service

    def test_retry_drops_context_and_cleans_markers(self):
        """재시도 시 문맥이 제거되고 생성 결과의 표식·따옴표가 정리되는지 확인한다."""
        generate = Mock(side_effect=['English only', '### Response:\n출력: "한국어 번역."'])
        with patch.dict(sys.modules, {'mlx_lm': SimpleNamespace(generate=generate)}):
            result = self.translator().translate('Hello.', '이전 문맥', 128)
        self.assertEqual(result['translation'], '한국어 번역.')
        self.assertIn('이전 문맥', generate.call_args_list[0].kwargs['prompt'])
        self.assertNotIn('이전 문맥', generate.call_args_list[1].kwargs['prompt'])

    def test_two_invalid_outputs_fail(self):
        """두 번의 생성 결과가 모두 유효하지 않으면 검증 오류가 발생하는지 확인한다."""
        generate = Mock(return_value='English only')
        with patch.dict(sys.modules, {'mlx_lm': SimpleNamespace(generate=generate)}):
            with self.assertRaises(ValueError):
                self.translator().translate('Hello.', '', 128)
        self.assertEqual(generate.call_count, 2)

    def test_token_limit_prevents_generation(self):
        """토큰 제한을 넘는 요청이 모델 생성 전에 거부되는지 확인한다."""
        generate = Mock()
        with patch.dict(sys.modules, {'mlx_lm': SimpleNamespace(generate=generate)}):
            with self.assertRaises(ValueError):
                self.translator().translate('x' * 2048, '', 128)
        generate.assert_not_called()

    def test_invalid_audio_before_model_loading(self):
        """손상·빈 오디오와 비정상 샘플이 모델 로딩 전에 거부되는지 확인한다."""
        service = TranscriptionService()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.wav'
            path.write_bytes(b'not audio')
            with self.assertRaisesRegex(ValueError, 'decodable audio'):
                service.transcribe(path)
        for audio in (np.array([], dtype=np.float32), np.array([np.nan]), np.array([np.inf])):
            with patch('faster_whisper.audio.decode_audio', return_value=audio):
                with self.assertRaises(ValueError):
                    service.transcribe(Path('invalid.wav'))
        self.assertFalse(service.loaded)

    def test_segments_stay_within_audio_duration(self):
        """자막 구간을 오디오 길이 안으로 보정하고 범위 밖 구간을 제거하는지 확인한다."""
        service = TranscriptionService()
        service._model = SimpleNamespace(transcribe=Mock(return_value=(iter([
            SimpleNamespace(start=-0.2, end=1.5, text=' Hello. '),
            SimpleNamespace(start=2.0, end=3.0, text='Outside.')]),
            SimpleNamespace(language='en', language_probability=1.0))))
        with patch('faster_whisper.audio.decode_audio', return_value=np.zeros(16000)):
            result = service.transcribe(Path('clip.wav'))
        self.assertEqual(result['segments'], [{'start': 0.0, 'end': 1.0, 'text': 'Hello.'}])


if __name__ == '__main__':
    unittest.main()
