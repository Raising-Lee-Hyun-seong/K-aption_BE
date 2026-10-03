"""모델 로딩 없이 실제 ASGI 요청·응답과 OpenAPI 자막 계약을 검증한다."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.main import app
from app.schemas import TranscriptionResponse


class ApiContractTests(unittest.IsolatedAsyncioTestCase):
    """HTTP 상태, 요청 검증 및 자막 응답 직렬화를 확인한다."""

    async def request(self, path, body=b'', query=b'', headers=None):
        """ASGI 앱에 HTTP 요청을 전달하고 상태 코드와 JSON 응답을 반환한다."""
        messages = []
        sent = False

        async def receive():
            """요청 본문을 한 번 전달하고 이후에는 연결 종료를 알린다."""
            nonlocal sent
            if not sent:
                sent = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            return {'type': 'http.disconnect'}

        async def send(message):
            """앱이 전송한 HTTP 응답 이벤트를 수집한다."""
            messages.append(message)

        await app({'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                   'method': 'POST', 'scheme': 'http', 'path': path, 'raw_path': path.encode(),
                   'query_string': query, 'headers': headers or [],
                   'client': ('127.0.0.1', 1234), 'server': ('test', 80), 'root_path': ''},
                  receive, send)
        status = next(m['status'] for m in messages if m['type'] == 'http.response.start')
        payload = b''.join(m.get('body', b'') for m in messages if m['type'] == 'http.response.body')
        return status, json.loads(payload)

    async def upload(self, translator, filename='lecture.wav', query=b'', stt_error=None):
        """multipart 업로드 요청을 테스트용 서비스로 처리한다."""
        body = (f'--boundary\r\nContent-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\naudio\r\n'
                '--boundary--\r\n').encode()
        stt = SimpleNamespace(transcribe=Mock(return_value={'segments': [
            {'start': 0, 'end': 1, 'text': 'Hello.'}]} , side_effect=stt_error))
        with patch.object(app.state, 'transcriber', stt, create=True), patch.object(
                app.state, 'translator', translator, create=True):
            return await self.request('/api/v1/transcriptions', body, query,
                                      [(b'content-type', b'multipart/form-data; boundary=boundary')])

    async def test_success_and_partial_failure_return_200(self):
        """정상 번역과 번역 검증 실패가 HTTP 200에서 구분 가능한 JSON으로 반환되는지 확인한다."""
        for fail in (False, True):
            translator = SimpleNamespace(translate=Mock(return_value={'translation': '안녕하세요.'},
                                                        side_effect=ValueError('bad') if fail else None))
            status, body = await self.upload(translator)
            self.assertEqual(status, 200)
            segment = body['segments'][0]
            self.assertEqual(segment['translation'], None if fail else '안녕하세요.')
            self.assertEqual(segment['translation_error'], 'translation_failed' if fail else None)
            TranscriptionResponse.model_validate(body)

    async def test_request_and_media_errors(self):
        """필수 파일·언어·확장자·오디오 오류가 문서화된 HTTP 상태로 반환되는지 확인한다."""
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
        """Swagger의 정상·실패·빈 결과 예시와 영어 언어 제한이 실제 계약과 일치하는지 검증한다."""
        spec = app.openapi()
        schema = spec['components']['schemas']['TranscriptionResponse']
        for example in schema['examples']:
            TranscriptionResponse.model_validate(example)
        operation = spec['paths']['/api/v1/transcriptions']['post']
        language = next(p for p in operation['parameters'] if p['name'] == 'language')
        self.assertEqual(language['schema']['const'], 'en')
        self.assertTrue({'200', '415', '422', '500'} <= operation['responses'].keys())
