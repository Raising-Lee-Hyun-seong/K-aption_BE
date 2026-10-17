"""모델 없이 업로드 입력 규칙 조회의 HTTP·스키마·검증 상수 일치를 확인한다."""

import json
import unittest
from unittest.mock import patch

from app.main import app
from app.schemas import CapabilitiesResponse
from app.services.media import ALLOWED_SUFFIXES, MAX_MEDIA_SECONDS, MAX_UPLOAD_BYTES


class CapabilitiesTests(unittest.IsolatedAsyncioTestCase):
    """프론트엔드가 실제 서버 제한을 안정적인 공개 응답으로 조회하는지 검증한다."""

    async def request(self, method='GET', origin=None):
        """모델 수명주기 실행 없이 실제 ASGI 입력 규칙 API를 호출한다.

        :param method: GET 또는 허용 여부를 확인할 HTTP 메서드.
        :param origin: CORS 헤더 검증에 사용할 선택적 브라우저 출처 문자열.
        :return: 상태 코드·JSON 본문·소문자 바이트 헤더 사전 튜플.
        """
        messages = []

        async def receive():
            """본문 없는 HTTP 요청 이벤트를 제공한다.

            :return: ASGI 요청 이벤트.
            """
            return {'type': 'http.request', 'body': b'', 'more_body': False}

        async def send(message):
            """앱의 HTTP 응답 이벤트를 기록한다.

            :param message: 수집할 ASGI 응답 이벤트.
            :return: None.
            """
            messages.append(message)

        path = '/api/v1/capabilities'
        headers = [(b'origin', origin.encode())] if origin else []
        await app({'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                   'method': method, 'scheme': 'http', 'path': path, 'raw_path': path.encode(),
                   'query_string': b'', 'headers': headers, 'client': ('127.0.0.1', 1234),
                   'server': ('test', 80), 'root_path': ''}, receive, send)
        start = next(message for message in messages if message['type'] == 'http.response.start')
        payload = b''.join(message.get('body', b'') for message in messages
                           if message['type'] == 'http.response.body')
        return start['status'], json.loads(payload), dict(start['headers'])

    async def test_response_matches_actual_server_limits(self):
        """실제 검증 상수와 동일한 제한을 명시적인 단위와 정렬된 확장자로 제공한다.

        :return: None.
        """
        status, body, headers = await self.request()
        self.assertEqual(status, 200)
        parsed = CapabilitiesResponse.model_validate(body)
        self.assertEqual(parsed.upload.allowed_extensions, sorted(ALLOWED_SUFFIXES))
        self.assertEqual(parsed.upload.max_upload_bytes, MAX_UPLOAD_BYTES)
        self.assertEqual(parsed.upload.max_duration_seconds, MAX_MEDIA_SECONDS)
        self.assertEqual(body, {'upload': {
            'allowed_extensions': ['.m4a', '.mkv', '.mov', '.mp3', '.mp4', '.wav', '.webm'],
            'max_upload_bytes': 1073741824, 'max_duration_seconds': 5400,
            'supported_languages': ['en'],
        }})
        self.assertEqual(headers[b'content-type'], b'application/json')
        self.assertEqual(headers[b'cache-control'], b'no-store')

    async def test_request_needs_no_models_jobs_or_file_storage(self):
        """미준비 서비스 상태에서도 모델·작업자 생성이나 파일 저장 없이 조회한다.

        :return: None.
        """
        with patch.dict(app.state._state, {'translator': None, 'transcriber': None, 'jobs': None}), \
                patch('app.main.TranslationService') as translator, \
                patch('app.main.TranscriptionService') as transcriber, \
                patch('app.main.JobManager') as manager, \
                patch('app.main.tempfile.NamedTemporaryFile') as temporary:
            status, _, _ = await self.request()
        self.assertEqual(status, 200)
        for dependency in (translator, transcriber, manager, temporary):
            dependency.assert_not_called()

    async def test_browser_cors_keeps_existing_origin_restriction(self):
        """기존 허용 출처에는 CORS 헤더를 제공하고 임의 출처에는 제공하지 않는다.

        :return: None.
        """
        from app.main import origins
        for origin in origins:
            status, _, headers = await self.request(origin=origin)
            self.assertEqual(status, 200)
            self.assertEqual(headers[b'access-control-allow-origin'].decode(), origin)
        status, _, headers = await self.request(origin='https://untrusted.invalid')
        self.assertEqual(status, 200)
        self.assertNotIn(b'access-control-allow-origin', headers)

    async def test_post_cannot_change_upload_rules(self):
        """조회 API에 POST를 보내도 서버 입력 규칙을 수정하지 못한다.

        :return: None.
        """
        status, _, headers = await self.request(method='POST')
        self.assertEqual(status, 405)
        self.assertIn(b'GET', headers[b'allow'])

    async def test_openapi_documents_units_language_and_response(self):
        """OpenAPI에 업로드 제한의 단위·포함 경계·입력 언어 계약을 노출한다.

        :return: None.
        """
        spec = app.openapi()
        operation = spec['paths']['/api/v1/capabilities']['get']
        self.assertEqual(operation['tags'], ['System'])
        self.assertEqual(operation['responses']['200']['content']['application/json']['schema'],
                         {'$ref': '#/components/schemas/CapabilitiesResponse'})
        fields = spec['components']['schemas']['UploadCapabilities']['properties']
        self.assertIn('바이트', fields['max_upload_bytes']['description'])
        self.assertIn('상한 포함', fields['max_duration_seconds']['description'])
        self.assertEqual(fields['supported_languages']['items']['const'], 'en')
