"""모델 로딩 없이 실제 ASGI 요청·응답과 OpenAPI 자막 계약을 검증한다."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.main import app
from app.schemas import TranscriptionResponse
from timing import logger as timing_logger


class ApiContractTests(unittest.IsolatedAsyncioTestCase):
    """HTTP 상태, 요청 검증 및 자막 응답 직렬화를 확인한다."""

    async def request(self, path, body=b'', query=b'', headers=None):
        """ASGI 앱에 HTTP 요청을 전달하고 상태 코드와 JSON 응답을 반환한다.

        :param path: 호출할 HTTP 요청 경로.
        :param body: ASGI 요청에 전달할 본문 바이트.
        :param query: URL 인코딩한 쿼리 문자열 바이트.
        :param headers: 선택적 ASGI 헤더 이름·값 바이트 쌍 목록.
        :return: HTTP 상태 코드와 디코딩한 JSON 응답의 튜플.
        """
        messages = []
        sent = False

        async def receive():
            """요청 본문을 한 번 전달하고 이후에는 연결 종료를 알린다.

            :return: 요청 본문 또는 연결 종료를 나타내는 ASGI 이벤트 사전.
            """
            nonlocal sent
            if not sent:
                sent = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            return {'type': 'http.disconnect'}

        async def send(message):
            """앱이 전송한 HTTP 응답 이벤트를 수집한다.

            :param message: 전송하거나 수집할 ASGI 응답 이벤트 사전.
            :return: None.
            """
            messages.append(message)

        await app({'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                   'method': 'POST', 'scheme': 'http', 'path': path, 'raw_path': path.encode(),
                   'query_string': query, 'headers': headers or [],
                   'client': ('127.0.0.1', 1234), 'server': ('test', 80), 'root_path': ''},
                  receive, send)
        status = next(m['status'] for m in messages if m['type'] == 'http.response.start')
        payload = b''.join(m.get('body', b'') for m in messages if m['type'] == 'http.response.body')
        return status, json.loads(payload)

    async def upload(self, translator, filename='lecture.wav', query=b'', stt_error=None,
                     segments=None):
        """multipart 업로드 요청을 선택한 STT 원문과 테스트용 서비스로 처리한다.

        :param translator: 테스트에서 사용할 번역 서비스 대역.
        :param filename: 테스트 업로드 파일 이름. 확장자가 미디어 허용 여부에 사용된다.
        :param query: URL 인코딩한 쿼리 문자열 바이트.
        :param stt_error: 음성 인식 대역이 발생시킬 선택적 예외.
        :param segments: 시작·종료 시각과 text를 담은 STT 구간 목록.
        :return: HTTP 상태 코드와 자막 JSON 응답의 튜플.
        """
        body = (f'--boundary\r\nContent-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\naudio\r\n'
                '--boundary--\r\n').encode()
        if segments is None:
            segments = [{'start': 0, 'end': 1, 'text': 'Hello.'}]
        stt = SimpleNamespace(transcribe=Mock(return_value={'segments': segments},
                                            side_effect=stt_error))
        with patch.object(app.state, 'transcriber', stt, create=True), patch.object(
                app.state, 'translator', translator, create=True):
            return await self.request('/api/v1/transcriptions', body, query,
                                      [(b'content-type', b'multipart/form-data; boundary=boundary')])

    async def test_success_and_partial_failure_return_200(self):
        """정상 번역과 번역 검증 실패가 HTTP 200에서 구분 가능한 JSON으로 반환되는지 확인한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        for fail in (False, True):
            translator = SimpleNamespace(translate=Mock(return_value={'translation': '안녕하세요.'},
                                                        side_effect=ValueError('bad') if fail else None))
            status, body = await self.upload(translator)
            self.assertEqual(status, 200)
            segment = body['segments'][0]
            self.assertEqual(segment['translation'], None if fail else '안녕하세요.')
            self.assertEqual(segment['translation_error'], 'translation_failed' if fail else None)
            TranscriptionResponse.model_validate(body)

    async def test_partial_failure_logs_request_and_batch_summary(self):
        """실제 요청 경로에서 구간 실패 수와 HTTP 200 및 동일 요청 식별자를 기록한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        translator = SimpleNamespace(translate=Mock(side_effect=ValueError('bad output')))
        with self.assertLogs(timing_logger, level='INFO') as logs:
            status, body = await self.upload(translator)
        records = [record.timing for record in logs.records]
        self.assertEqual(status, 200)
        batch = next(record for record in records if record['event'] == 'translation_batch')
        segment = next(record for record in records if record['event'] == 'segment_translation')
        self.assertEqual(batch['failed_segments'], 1)
        self.assertEqual(batch['source_segment_count'], 1)
        self.assertEqual(batch['sentence_count'], 1)
        self.assertEqual(segment['segment_index'], 0)
        self.assertEqual(segment['status'], 'error')
        self.assertEqual(records[-1]['event'], 'http_request')
        self.assertEqual(records[-1]['status_code'], 200)
        self.assertEqual(len({record['request_id'] for record in records}), 1)

    async def test_grouped_subtitle_contract_and_timing_counts(self):
        """문장 묶기 후에도 기존 JSON 필드를 유지하고 원본·반환 구간 수를 로그로 구분한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        segments = [{'start': 0, 'end': 1, 'text': 'The mass is'},
                    {'start': 1, 'end': 2, 'text': 'constant.'},
                    {'start': 2, 'end': 3, 'text': 'Next.'}]
        translator = SimpleNamespace(translate=Mock(side_effect=[
            {'translation': '질량은 일정합니다.'}, ValueError('bad output')]))
        with self.assertLogs(timing_logger, level='INFO') as logs:
            status, body = await self.upload(translator, segments=segments)
        self.assertEqual(status, 200)
        self.assertEqual(len(body['segments']), 2)
        self.assertEqual(body['segments'][0], {
            'start': 0.0, 'end': 2.0, 'text': 'The mass is constant.',
            'translation': '질량은 일정합니다.', 'translation_error': None})
        self.assertEqual(set(body['segments'][1]),
                         {'start', 'end', 'text', 'translation', 'translation_error'})
        records = [record.timing for record in logs.records]
        batch = next(record for record in records if record['event'] == 'translation_batch')
        self.assertEqual(batch['source_segment_count'], 3)
        self.assertEqual(batch['sentence_count'], 2)
        self.assertEqual(batch['failed_segments'], 1)
        self.assertEqual([record['segment_index'] for record in records
                          if record['event'] == 'segment_translation'], [0, 1])

    async def test_request_and_media_errors(self):
        """필수 파일·언어·확장자·오디오 오류가 문서화된 HTTP 상태로 반환되는지 확인한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        translator = SimpleNamespace(translate=Mock())
        status, body = await self.request('/api/v1/transcriptions')
        self.assertEqual(status, 422)
        self.assertIsInstance(body['detail'], list)
        status, body = await self.upload(translator, query=b'language=ko')
        self.assertEqual(status, 422)
        self.assertIsInstance(body['detail'], list)
        status, body = await self.upload(translator, filename='lecture.txt')
        self.assertEqual(status, 415)
        self.assertEqual(body['detail'], 'Unsupported media file extension')
        status, body = await self.upload(translator, stt_error=ValueError('bad audio'))
        self.assertEqual(status, 422)
        self.assertEqual(body['detail'], 'bad audio')
        translator.translate.assert_not_called()

    async def test_openapi_examples_match_schema(self):
        """Swagger의 정상·실패·빈 결과 예시와 영어 언어 제한이 실제 계약과 일치하는지 검증한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        spec = app.openapi()
        schema = spec['components']['schemas']['TranscriptionResponse']
        for example in schema['examples']:
            TranscriptionResponse.model_validate(example)
        operation = spec['paths']['/api/v1/transcriptions']['post']
        language = next(p for p in operation['parameters'] if p['name'] == 'language')
        self.assertEqual(language['schema']['const'], 'en')
        self.assertTrue({'200', '415', '422', '500'} <= operation['responses'].keys())
