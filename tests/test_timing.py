"""실제 모델 실행 없이 작업 시간·오류 로그와 동시 API 요청의 문맥 격리를 검증한다."""
import asyncio
import unittest
from unittest.mock import patch

from starlette.concurrency import run_in_threadpool

import timing
from app.middleware import TimingMiddleware


class TimingTests(unittest.TestCase):
    """단조 시계로 계산한 시간과 안전한 메타데이터 및 예외 보존을 검증한다."""

    def test_measure_records_elapsed_and_mutable_metadata(self):
        """본문에서 추가한 구간 수와 요청 식별자를 정확한 작업 시간과 함께 기록한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        request_token = timing.request_id.set('request-one')
        segment_token = timing.segment_index.set(3)
        try:
            with patch('timing.perf_counter', side_effect=[12.0, 14.25]):
                with self.assertLogs(timing.logger, level='INFO') as logs:
                    with timing.measure('stt', language='en') as metadata:
                        metadata['segment_count'] = 5
            self.assertEqual(len(logs.records), 1)
            self.assertEqual(logs.records[0].timing, {
                'event': 'stt', 'request_id': 'request-one', 'segment_index': 3,
                'elapsed_seconds': 2.25, 'status': 'ok', 'language': 'en',
                'segment_count': 5,
            })
            self.assertIn('elapsed_seconds=2.250', logs.records[0].getMessage())
        finally:
            timing.segment_index.reset(segment_token)
            timing.request_id.reset(request_token)

    def test_errors_and_cancellation_are_logged_without_exception_content(self):
        """일반 예외와 취소 모두 시간·예외 종류만 남기고 원래 예외 객체를 전달한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        for exception in (ValueError('private original text'),
                          asyncio.CancelledError('private translation text')):
            with self.subTest(exception=type(exception).__name__):
                with patch('timing.perf_counter', side_effect=[40.0, 40.5]):
                    with self.assertLogs(timing.logger, level='INFO') as logs:
                        with self.assertRaises(type(exception)) as raised:
                            with timing.measure('translation_generation', attempt=2):
                                raise exception
                self.assertIs(raised.exception, exception)
                metadata = logs.records[0].timing
                self.assertEqual(metadata['elapsed_seconds'], 0.5)
                self.assertEqual(metadata['status'], 'error')
                self.assertEqual(metadata['error_type'], type(exception).__name__)
                self.assertNotIn('private', repr(metadata))
                self.assertNotIn('private', logs.records[0].getMessage())
                self.assertIsNone(logs.records[0].exc_info)

    def test_log_duration_records_only_supplied_metadata_and_context(self):
        """시간 로그가 기본 문맥과 제공된 개수만 포함하고 원문·번역·프롬프트를 만들지 않는다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        request_token = timing.request_id.set('-')
        segment_token = timing.segment_index.set(None)
        try:
            with self.assertLogs(timing.logger, level='INFO') as logs:
                timing.log_duration('translation_summary', 1.75,
                                    segment_count=4, failed_count=1)
            self.assertEqual(logs.records[0].timing, {
                'event': 'translation_summary', 'request_id': '-',
                'segment_index': None, 'elapsed_seconds': 1.75,
                'segment_count': 4, 'failed_count': 1,
            })
        finally:
            timing.segment_index.reset(segment_token)
            timing.request_id.reset(request_token)


class TimingMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    """응답 상태·시간 및 동시 요청 간 식별자와 구간 문맥의 복구를 검증한다."""

    def scope(self, path='/api/v1/transcriptions', kind='http'):
        """미들웨어 검사에 필요한 최소 ASGI 요청 문맥을 만든다.

        :param path: 테스트 ASGI 요청 경로.
        :param kind: 테스트 요청의 ASGI scope 종류.
        :return: 지정 경로·종류를 포함하는 ASGI 문맥 사전.
        """
        return {'type': kind, 'path': path, 'method': 'POST',
                'headers': [(b'x-request-id', b'untrusted-external-id')]}

    async def run_request(self, middleware, scope=None):
        """요청 본문을 제공하고 미들웨어가 전송한 ASGI 응답 이벤트를 반환한다.

        :param middleware: 테스트할 요청 시간 측정 미들웨어.
        :param scope: 요청 종류·경로·헤더 등을 담은 ASGI 문맥.
        :return: 미들웨어가 전송한 ASGI 응답 이벤트 목록.
        """
        messages = []

        async def receive():
            """테스트용 빈 HTTP 본문을 제공한다.

            :return: 빈 본문과 more_body=False를 담은 http.request 이벤트 사전.
            """
            return {'type': 'http.request', 'body': b'', 'more_body': False}

        async def send(message):
            """전송된 응답 이벤트를 수집한다.

            :param message: 전송하거나 수집할 ASGI 응답 이벤트 사전.
            :return: None.
            """
            messages.append(message)

        await middleware(scope or self.scope(), receive, send)
        return messages

    async def test_validation_response_logs_422_and_restores_context(self):
        """검증 실패 응답의 실제 상태와 시간을 기록하고 외부 문맥·응답을 보존한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        observed = []

        async def app(scope, receive, send):
            """새 요청 문맥을 관찰하고 검증 오류 응답을 전송한다.

            :param scope: 요청 종류·경로·헤더 등을 담은 ASGI 문맥.
            :param receive: ASGI 요청 이벤트를 비동기로 수신하는 함수.
            :param send: ASGI 응답 이벤트를 비동기로 전송하는 함수.
            :return: None.
            """
            observed.append((timing.request_id.get(), timing.segment_index.get()))
            await send({'type': 'http.response.start', 'status': 422, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'validation error'})

        request_token = timing.request_id.set('outer-request')
        segment_token = timing.segment_index.set(7)
        try:
            with patch('app.middleware.perf_counter', side_effect=[4.0, 6.5]):
                with self.assertLogs(timing.logger, level='INFO') as logs:
                    messages = await self.run_request(TimingMiddleware(app))
            self.assertEqual(timing.request_id.get(), 'outer-request')
            self.assertEqual(timing.segment_index.get(), 7)
            self.assertEqual(messages, [
                {'type': 'http.response.start', 'status': 422, 'headers': []},
                {'type': 'http.response.body', 'body': b'validation error'},
            ])
            self.assertEqual([record.timing['event'] for record in logs.records],
                             ['request_started', 'http_request'])
            start, end = [record.timing for record in logs.records]
            self.assertEqual(start['request_id'], observed[0][0])
            self.assertNotEqual(start['request_id'], 'untrusted-external-id')
            self.assertEqual(observed[0][1], None)
            self.assertEqual(end['request_id'], start['request_id'])
            self.assertEqual(end['elapsed_seconds'], 2.5)
            self.assertEqual(end['status_code'], 422)
            self.assertEqual(end['status'], 'error')
        finally:
            timing.segment_index.reset(segment_token)
            timing.request_id.reset(request_token)

    async def test_unhandled_error_logs_500_and_restores_context(self):
        """응답 전 예외를 500으로 기록하되 원래 예외를 전달하고 문맥을 복구한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        exception = RuntimeError('private prompt content')

        async def app(scope, receive, send):
            """응답을 시작하기 전에 테스트용 추론 오류를 발생시킨다.

            :param scope: 요청 종류·경로·헤더 등을 담은 ASGI 문맥.
            :param receive: ASGI 요청 이벤트를 비동기로 수신하는 함수.
            :param send: ASGI 응답 이벤트를 비동기로 전송하는 함수.
            :return: None. 정상 반환하지 않는다.
            :raises RuntimeError: 테스트용 추론 오류를 항상 발생시킨다.
            """
            raise exception

        request_token = timing.request_id.set('outer-request')
        segment_token = timing.segment_index.set(8)
        try:
            with patch('app.middleware.perf_counter', side_effect=[7.0, 9.0]):
                with self.assertLogs(timing.logger, level='INFO') as logs:
                    with self.assertRaises(RuntimeError) as raised:
                        await self.run_request(TimingMiddleware(app))
            self.assertIs(raised.exception, exception)
            metadata = logs.records[-1].timing
            self.assertEqual(metadata['status_code'], 500)
            self.assertEqual(metadata['elapsed_seconds'], 2.0)
            self.assertEqual(metadata['error_type'], 'RuntimeError')
            self.assertNotIn('private', repr(metadata))
            self.assertNotIn('private', logs.records[-1].getMessage())
            self.assertEqual(timing.request_id.get(), 'outer-request')
            self.assertEqual(timing.segment_index.get(), 8)
        finally:
            timing.segment_index.reset(segment_token)
            timing.request_id.reset(request_token)

    async def test_concurrent_requests_keep_distinct_ids_and_segments(self):
        """동시 요청과 추론 스레드풀에서 식별자·구간 번호가 섞이지 않고 문맥이 유지된다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        entered = asyncio.Event()
        observations = {}

        def inference(path, expected_id, expected_segment):
            """추론 스레드에 전달된 요청·구간 문맥을 확인하고 작업 시간 로그를 남긴다.

            :param path: 시간 로그에 기록할 요청 경로.
            :param expected_id: 추론 스레드에 전달되어야 하는 요청 식별자.
            :param expected_segment: 추론 스레드에 전달되어야 하는 구간 인덱스.
            :return: None.
            """
            self.assertEqual(timing.request_id.get(), expected_id)
            self.assertEqual(timing.segment_index.get(), expected_segment)
            timing.log_duration('translation_generation', 0.1, path=path)

        async def app(scope, receive, send):
            """두 요청이 겹치도록 기다린 뒤 각 요청의 작업 로그와 응답을 남긴다.

            :param scope: 요청 종류·경로·헤더 등을 담은 ASGI 문맥.
            :param receive: ASGI 요청 이벤트를 비동기로 수신하는 함수.
            :param send: ASGI 응답 이벤트를 비동기로 전송하는 함수.
            :return: None.
            """
            path = scope['path']
            current_id = timing.request_id.get()
            self.assertEqual(timing.segment_index.get(), None)
            current_segment = 1 if path.endswith('first') else 2
            segment_token = timing.segment_index.set(current_segment)
            try:
                observations[path] = current_id
                if len(observations) == 2:
                    entered.set()
                await entered.wait()
                await asyncio.sleep(0)
                self.assertEqual(timing.request_id.get(), current_id)
                self.assertEqual(timing.segment_index.get(), current_segment)
                await run_in_threadpool(inference, path, current_id, current_segment)
            finally:
                timing.segment_index.reset(segment_token)
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'ok'})

        request_token = timing.request_id.set('outer-request')
        segment_token = timing.segment_index.set(9)
        try:
            middleware = TimingMiddleware(app)
            with self.assertLogs(timing.logger, level='INFO') as logs:
                await asyncio.wait_for(asyncio.gather(
                    self.run_request(middleware, self.scope('/api/v1/first')),
                    self.run_request(middleware, self.scope('/api/v1/second')),
                ), timeout=2)
            self.assertEqual(len(set(observations.values())), 2)
            for record in logs.records:
                metadata = record.timing
                path = metadata['path']
                self.assertEqual(metadata['request_id'], observations[path])
                if metadata['event'] == 'translation_generation':
                    self.assertEqual(metadata['segment_index'],
                                     1 if path.endswith('first') else 2)
                else:
                    self.assertEqual(metadata['segment_index'], None)
            completed = [record.timing for record in logs.records
                         if record.timing['event'] == 'http_request']
            self.assertEqual(len(completed), 2)
            self.assertTrue(all(record['status_code'] == 200 for record in completed))
            self.assertEqual(timing.request_id.get(), 'outer-request')
            self.assertEqual(timing.segment_index.get(), 9)
        finally:
            timing.segment_index.reset(segment_token)
            timing.request_id.reset(request_token)

    async def test_non_api_and_non_http_requests_pass_through(self):
        """문서·헬스·비HTTP 요청은 시간 로그나 새 요청 문맥 없이 하위 앱으로 전달한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        seen = []

        async def app(scope, receive, send):
            """미들웨어를 통과한 요청 종류와 기존 문맥을 수집한다.

            :param scope: 요청 종류·경로·헤더 등을 담은 ASGI 문맥.
            :param receive: ASGI 요청 이벤트를 비동기로 수신하는 함수.
            :param send: ASGI 응답 이벤트를 비동기로 전송하는 함수.
            :return: None.
            """
            seen.append((scope['type'], scope['path'], timing.request_id.get(),
                         timing.segment_index.get()))

        request_token = timing.request_id.set('outer-request')
        segment_token = timing.segment_index.set(11)
        try:
            with patch.object(timing.logger, 'info') as log_info:
                for path, kind in (('/docs', 'http'), ('/health', 'http'),
                                   ('/api/v1/socket', 'websocket')):
                    await self.run_request(TimingMiddleware(app), self.scope(path, kind))
                log_info.assert_not_called()
            self.assertEqual(seen, [
                ('http', '/docs', 'outer-request', 11),
                ('http', '/health', 'outer-request', 11),
                ('websocket', '/api/v1/socket', 'outer-request', 11),
            ])
        finally:
            timing.segment_index.reset(segment_token)
            timing.request_id.reset(request_token)


if __name__ == '__main__':
    unittest.main()
