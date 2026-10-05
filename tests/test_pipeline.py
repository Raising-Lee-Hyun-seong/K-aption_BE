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
from timing import logger as timing_logger


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    """비동기 STT·번역 엔드포인트의 응답과 자원 정리를 검증한다."""
    async def run_upload(self, transcriber, translator, filename='lecture.wav'):
        """테스트 업로드를 처리하고 파일 종료와 임시 파일 삭제를 확인한다.

        :param transcriber: 테스트에서 사용할 음성 인식 서비스 대역.
        :param translator: 테스트에서 사용할 번역 서비스 대역.
        :param filename: 테스트 업로드 파일 이름. 확장자가 미디어 허용 여부에 사용된다.
        :return: 업로드 처리의 TranscriptionResponse. 예외가 나도 파일 정리를 확인한다.
        """
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

    async def test_segments_preserve_timestamps_and_previous_source(self):
        """구간 순서·타임스탬프 보존과 직전 영어 원문의 문맥 전달을 검증한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
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
        self.assertEqual(translator.translate.call_args_list[1].args, ('Second.', 'First.', 128))

    async def test_fragments_are_grouped_before_translation(self):
        """끊긴 문장을 한 번 번역하고 합친 시각을 반환하며 내부 원본 인덱스는 노출하지 않는다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        segments = [{'start': 1.0, 'end': 2.0, 'text': 'Separate the'},
                    {'start': 2.0, 'end': 3.5, 'text': 'variables.'},
                    {'start': 3.5, 'end': 5.0, 'text': 'Then integrate.'}]
        translator = SimpleNamespace(translate=Mock(side_effect=[
            {'translation': '변수를 분리합니다.'}, {'translation': '그런 다음 적분합니다.'}]))
        result = await self.run_upload(SimpleNamespace(transcribe=Mock(
            return_value={'segments': segments})), translator)
        self.assertEqual(result.model_dump(), {'segments': [
            {'start': 1.0, 'end': 3.5, 'text': 'Separate the variables.',
             'translation': '변수를 분리합니다.', 'translation_error': None},
            {'start': 3.5, 'end': 5.0, 'text': 'Then integrate.',
             'translation': '그런 다음 적분합니다.', 'translation_error': None},
        ]})
        self.assertEqual(translator.translate.call_args_list[0].args,
                         ('Separate the variables.', '', 128))
        self.assertEqual(translator.translate.call_args_list[1].args,
                         ('Then integrate.', 'Separate the variables.', 128))

    async def test_previous_source_context_is_bounded(self):
        """긴 앞 문장에서도 마지막 256자 영어 원문만 문맥으로 전달한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        source = 'First fragment ' * 30 + '.'
        segments = [{'start': 0, 'end': 10, 'text': source},
                    {'start': 10, 'end': 11, 'text': 'Next.'}]
        translator = SimpleNamespace(translate=Mock(return_value={'translation': '번역.'}))
        await self.run_upload(SimpleNamespace(transcribe=Mock(
            return_value={'segments': segments})), translator)
        self.assertEqual(translator.translate.call_args_list[1].args[1], source[-256:])

    async def test_errors_cleanup_upload_and_temp_file(self):
        """STT 검증 오류와 예기치 않은 번역 오류에서도 파일이 정리되는지 확인한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
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

    async def test_partial_failure_preserves_original_and_source_context(self):
        """번역 실패 구간의 원문을 보존하고 해당 영어 문맥으로 다음 구간을 계속 처리한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
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
        self.assertEqual(translator.translate.call_args_list[2].args[1], 'Sentence 1.')

    async def test_all_translations_fail_without_losing_stt(self):
        """모든 번역 검증이 실패해도 원문과 타임스탬프를 반환한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        segment = {'start': 0, 'end': 1, 'text': 'Hello.'}
        result = await self.run_upload(SimpleNamespace(transcribe=Mock(
            return_value={'segments': [segment]})), SimpleNamespace(
                translate=Mock(side_effect=ValueError('invalid output'))))
        self.assertEqual(result.model_dump(), {'segments': [{
            **segment, 'translation': None, 'translation_error': 'translation_failed'}]})

    async def test_unsupported_extension(self):
        """지원하지 않는 확장자를 추론 없이 HTTP 415로 거부하는지 확인한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        stt = SimpleNamespace(transcribe=Mock())
        with self.assertRaises(HTTPException) as error:
            await self.run_upload(stt, None, 'lecture.txt')
        self.assertEqual(error.exception.status_code, 415)
        stt.transcribe.assert_not_called()

    async def test_no_speech(self):
        """음성 구간이 없을 때 번역 호출 없이 빈 자막 목록을 반환하는지 확인한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        translator = SimpleNamespace(translate=Mock())
        result = await self.run_upload(SimpleNamespace(transcribe=Mock(
            return_value={'segments': []})), translator)
        self.assertEqual(result.segments, [])
        translator.translate.assert_not_called()


class ServiceTests(unittest.TestCase):
    """번역 재시도·토큰 제한과 실제 오디오 디코딩 검증을 수행한다."""
    def stream_mock(self, generate):
        """기존 텍스트 목을 MLX 응답 스트림으로 감싸 생성 호출을 관찰할 수 있게 한다.

        :param generate: 모델 생성 결과를 반환하는 테스트 대역.
        :return: MLX 응답 대역을 생성하는 responses 함수.
        """
        def responses(*args, **kwargs):
            """목 생성 결과를 종료 사유와 실제 생성 토큰 수가 있는 응답으로 반환한다.

            :param args: 생성 대역에 그대로 전달할 위치 인자.
            :param kwargs: 생성 대역에 그대로 전달할 키워드 인자.
            :return: 아래 값을 순차 제공하는 생성기.
            :yield: text·generation_tokens·finish_reason 속성을 가진 MLX 응답 대역.
            """
            yield SimpleNamespace(text=generate(*args, **kwargs), generation_tokens=17,
                                  finish_reason='stop')
        return responses

    def translator(self):
        """실제 모델 로딩 없이 번역 로직을 검증할 테스트용 서비스를 구성한다.

        :return: 모델을 로드하지 않는 TranslationService 테스트 인스턴스.
        """
        service = TranslationService.__new__(TranslationService)
        service.model = object()
        service.tokenizer = SimpleNamespace(encode=lambda text: list(text))
        service.template = {'prompt_input': '{instruction}\n{input}'}
        service.glossary = []
        service.sampler = object()
        service._lock = Lock()
        return service

    def test_retry_drops_context_and_cleans_markers(self):
        """재시도 시 문맥이 제거되고 생성 결과의 표식·따옴표가 정리되는지 확인한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        generate = Mock(side_effect=['English only', '### Response:\n출력: "한국어 번역."'])
        with patch.dict(sys.modules, {'mlx_lm': SimpleNamespace(stream_generate=self.stream_mock(generate))}):
            result = self.translator().translate('Hello.', '이전 문맥', 128)
        self.assertEqual(result['translation'], '한국어 번역.')
        self.assertIn('이전 문맥', generate.call_args_list[0].kwargs['prompt'])
        self.assertNotIn('이전 문맥', generate.call_args_list[1].kwargs['prompt'])

    def test_retry_logs_each_attempt_and_releases_model_lock(self):
        """대기·두 생성 시도·검증 결과·전체 시간과 잠금 해제를 함께 확인한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        service = self.translator()
        generate = Mock(side_effect=['English only', '한국어 번역.'])
        with patch.dict(sys.modules, {'mlx_lm': SimpleNamespace(stream_generate=self.stream_mock(generate))}):
            with self.assertLogs(timing_logger, level='INFO') as logs:
                service.translate('Private source.', '', 128)
        records = [record.timing for record in logs.records]
        attempts = [record for record in records if record['event'] == 'translation_attempt']
        self.assertEqual([record['accepted'] for record in attempts], [False, True])
        self.assertEqual([record['attempt'] for record in attempts], [1, 2])
        self.assertEqual(len([record for record in records if record['event'] == 'translation_generate']), 2)
        self.assertEqual(records[-1]['event'], 'translation_total')
        self.assertEqual(records[-1]['attempts'], 2)
        self.assertTrue(all(record['elapsed_seconds'] >= 0 for record in records))
        self.assertTrue(service._lock.acquire(blocking=False))
        service._lock.release()
        self.assertNotIn('Private source.', ' '.join(logs.output))
        self.assertNotIn('한국어 번역.', ' '.join(logs.output))

    def test_generation_error_logs_time_and_releases_model_lock(self):
        """예기치 않은 생성 오류도 시간을 남기며 다음 호출의 잠금을 막지 않는다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        service = self.translator()
        generate = Mock(side_effect=RuntimeError('private model output'))
        with patch.dict(sys.modules, {'mlx_lm': SimpleNamespace(stream_generate=self.stream_mock(generate))}):
            with self.assertLogs(timing_logger, level='INFO') as logs:
                with self.assertRaises(RuntimeError):
                    service.translate('Hello.', '', 128)
        self.assertEqual(logs.records[-1].timing['status'], 'error')
        self.assertEqual(logs.records[-1].timing['error_type'], 'RuntimeError')
        self.assertTrue(service._lock.acquire(blocking=False))
        service._lock.release()
        self.assertNotIn('private model output', ' '.join(logs.output))

    def test_two_invalid_outputs_fail(self):
        """두 번의 생성 결과가 모두 유효하지 않으면 검증 오류가 발생하는지 확인한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        generate = Mock(return_value='English only')
        with patch.dict(sys.modules, {'mlx_lm': SimpleNamespace(stream_generate=self.stream_mock(generate))}):
            with self.assertRaises(ValueError):
                self.translator().translate('Hello.', '', 128)
        self.assertEqual(generate.call_count, 2)

    def test_token_limit_prevents_generation(self):
        """토큰 제한을 넘는 요청이 모델 생성 전에 거부되는지 확인한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        generate = Mock()
        with patch.dict(sys.modules, {'mlx_lm': SimpleNamespace(stream_generate=self.stream_mock(generate))}):
            with self.assertRaises(ValueError):
                self.translator().translate('x' * 2048, '', 128)
        generate.assert_not_called()

    def test_glossary_retries_semantically_wrong_term(self):
        """알짜힘을 중력으로 번역한 결과를 거부하고 용어를 포함한 재시도 결과를 사용한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        service = self.translator()
        service.glossary = [{'en': 'net force', 'ko': '알짜힘'}]
        generate = Mock(side_effect=['물체의 중력이 없습니다.', '물체에 작용하는 알짜힘은 0입니다.'])
        with patch.dict(sys.modules, {'mlx_lm': SimpleNamespace(stream_generate=self.stream_mock(generate))}):
            result = service.translate('The net force is zero.', '', 128)
        self.assertIn('알짜힘', result['translation'])
        self.assertEqual(generate.call_count, 2)

    def test_context_copy_is_not_accepted_on_retry(self):
        """앞 구간과 같은 번역이 두 번 생성되면 실패로 처리한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        generate = Mock(return_value='속도는 일정합니다.')
        with patch.dict(sys.modules, {'mlx_lm': SimpleNamespace(stream_generate=self.stream_mock(generate))}):
            with self.assertRaises(ValueError):
                self.translator().translate('The mass is two kilograms.', '속도는 일정합니다.', 128)
        self.assertEqual(generate.call_count, 2)

    def test_glossary_matches_whole_words_and_clean_arrow(self):
        """용어의 단어 경계를 확인하고 화살표 뒤의 한국어 번역을 보존한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        service = self.translator()
        service.glossary = [{'en': 'mass', 'ko': '질량'}]
        self.assertEqual(service._matching_terms('MASS is constant.'), service.glossary)
        self.assertEqual(service._matching_terms('Massive object.'), [])
        self.assertEqual(service._clean_translation('Mass -> 질량'), '질량')

    def test_compound_term_does_not_require_nested_term(self):
        """복합 용어 구심력 안의 force에 별도 번역 힘을 중복 요구하지 않는다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        service = self.translator()
        compound = {'en': 'centripetal force', 'ko': '구심력'}
        simple = {'en': 'force', 'ko': '힘'}
        service.glossary = [simple, compound]
        self.assertEqual(service._matching_terms('Centripetal force acts.'), [compound])
        self.assertTrue(service._is_acceptable('구심력이 작용합니다.', 'Centripetal force acts.', ''))
        self.assertEqual(service._matching_terms('Centripetal force is a force.'), [compound, simple])

    def test_invalid_audio_before_model_loading(self):
        """손상·빈 오디오와 비정상 샘플이 모델 로딩 전에 거부되는지 확인한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
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
        """자막 구간을 오디오 길이 안으로 보정하고 범위 밖 구간을 제거하는지 확인한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
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
