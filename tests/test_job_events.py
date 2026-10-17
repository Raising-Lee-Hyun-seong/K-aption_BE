"""모델 추론 없이 작업 SSE의 상태 전달·종료·연결 해제·HTTP 계약을 검증한다."""

import asyncio
from datetime import datetime, timezone
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.main import app
from app.schemas import JobError, JobStatusResponse
from app.services import job_events as events


class JobEventsTests(unittest.IsolatedAsyncioTestCase):
    """SSE 생성기와 실제 ASGI 응답의 공개 계약을 확인한다."""

    def record(self, status='queued', **changes):
        """UTC 시각과 유효한 ID를 가진 작업 스냅샷을 만든다.

        :param status: queued·running·completed·failed 중 하나.
        :param changes: 기본 응답 필드에 덮어쓸 값.
        :return: 검증된 JobStatusResponse.
        """
        data = dict(job_id='a' * 32, status=status,
                    stage='transcribing' if status == 'running' else status,
                    created_at=datetime(2026, 10, 18, tzinfo=timezone.utc),
                    updated_at=datetime(2026, 10, 18, tzinfo=timezone.utc))
        data.update(changes)
        return JobStatusResponse(**data)

    def payload(self, chunk):
        """SSE status·unavailable 이벤트에서 JSON 본문을 추출한다.

        :param chunk: data 행을 포함한 SSE 문자열.
        :return: JSON으로 역직렬화한 사전.
        """
        return json.loads(next(line[6:] for line in chunk.splitlines() if line.startswith('data: ')))

    async def test_initial_progress_and_completion_use_status_schema(self):
        """접수·번역 진행·완료를 순서대로 보내고 완료 뒤 스트림을 끝낸다.

        :return: None.
        """
        running = self.record('running', stage='translating', total_segments=2,
                              completed_segments=1, progress_percent=50)
        done = self.record('completed', total_segments=2, completed_segments=2,
                           progress_percent=100)
        manager = SimpleNamespace(get=Mock(side_effect=[running, done]))
        with patch.object(events.asyncio, 'sleep', new_callable=AsyncMock):
            chunks = [chunk async for chunk in events.job_events(manager, self.record())]
        records = [JobStatusResponse.model_validate(self.payload(chunk)) for chunk in chunks]
        self.assertEqual([record.status for record in records], ['queued', 'running', 'completed'])
        self.assertEqual([record.progress_percent for record in records], [None, 50, 100])
        self.assertTrue(all(chunk.startswith('event: status\n') and chunk.endswith('\n\n')
                            for chunk in chunks))

    async def test_elapsed_time_alone_does_not_send_duplicate_status(self):
        """경과 시간만 증가하면 중복 이벤트를 생략하고 다음 상태 변경을 보낸다.

        :return: None.
        """
        manager = SimpleNamespace(get=Mock(side_effect=[
            self.record(elapsed_seconds=1), self.record('completed')]))
        with patch.object(events.asyncio, 'sleep', new_callable=AsyncMock):
            chunks = [chunk async for chunk in events.job_events(manager, self.record())]
        self.assertEqual(len(chunks), 2)
        self.assertEqual(manager.get.call_count, 2)

    async def test_completed_and_failed_jobs_emit_once_without_polling(self):
        """이미 종료된 작업은 실패 정보를 포함해 한 번만 보내고 조회를 추가하지 않는다.

        :return: None.
        """
        for state in ('completed', 'failed'):
            with self.subTest(state=state):
                error = JobError(code='processing_failed', message='처리에 실패했습니다.') if state == 'failed' else None
                initial = self.record(state, error=error)
                manager = SimpleNamespace(get=Mock())
                chunks = [chunk async for chunk in events.job_events(manager, initial)]
                self.assertEqual(len(chunks), 1)
                self.assertEqual(self.payload(chunks[0]), initial.model_dump(mode='json'))
                manager.get.assert_not_called()

    async def test_idle_stream_sends_heartbeat_comment(self):
        """변경 없는 연결에 15초가 지나면 JSON 상태 대신 heartbeat 주석을 보낸다.

        :return: None.
        """
        stream = events.job_events(SimpleNamespace(get=Mock()), self.record())
        with patch.object(events, 'monotonic', return_value=0) as clock:
            await stream.__anext__()
            clock.return_value = 15
            self.assertEqual(await stream.__anext__(), ': heartbeat\n\n')
            await stream.aclose()

    async def test_missing_or_storage_error_emits_safe_unavailable_and_closes(self):
        """스트리밍 도중 만료·저장 오류를 안전한 이벤트로 알리고 연결을 끝낸다.

        :return: None.
        """
        for error, code in ((KeyError('secret path'), 'job_not_found'),
                            (OSError('secret path'), 'storage_error')):
            with self.subTest(code=code), patch.object(events.asyncio, 'sleep', new_callable=AsyncMock):
                manager = SimpleNamespace(get=Mock(side_effect=error))
                chunks = [chunk async for chunk in events.job_events(manager, self.record())]
                self.assertEqual(len(chunks), 2)
                self.assertTrue(chunks[-1].startswith('event: unavailable\n'))
                self.assertEqual(self.payload(chunks[-1])['code'], code)
                self.assertNotIn('secret path', chunks[-1])

    async def test_cancelled_subscription_does_not_cancel_job(self):
        """대기 중 구독 태스크를 취소하면 취소를 전파하고 관리자 작업을 변경하지 않는다.

        :return: None.
        """
        manager = Mock()
        stream = events.job_events(manager, self.record())
        await stream.__anext__()
        pending = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        manager.assert_not_called()
        self.assertEqual(manager.method_calls, [])

    async def request(self, manager, job_id='a' * 32, disconnect=False):
        """SSE HTTP 요청을 앱에 전달하고 종료된 응답의 상태·헤더·본문을 모은다.

        :param manager: 요청 앱 상태에 주입할 작업 관리자 대역 또는 None.
        :param job_id: 요청 경로에 넣을 기존 또는 잘못된 작업 ID.
        :param disconnect: 첫 본문을 받은 직후 ASGI 연결 종료를 보낼지 여부.
        :return: HTTP 상태 코드·소문자 바이트 헤더 사전·본문 바이트 튜플.
        :raises TimeoutError: 종료돼야 하는 응답이 2초 안에 끝나지 않은 경우.
        """
        messages = []
        sent = False
        first_body = asyncio.Event()

        async def receive():
            """GET 본문을 한 번 제공한 뒤 연결을 유지한다.

            :return: ASGI 요청 이벤트.
            """
            nonlocal sent
            if not sent:
                sent = True
                return {'type': 'http.request', 'body': b'', 'more_body': False}
            if disconnect:
                await first_body.wait()
                return {'type': 'http.disconnect'}
            await asyncio.Event().wait()

        async def send(message):
            """실제 응답 이벤트를 순서대로 기록한다.

            :param message: ASGI 응답 이벤트.
            :return: None.
            """
            messages.append(message)
            if message['type'] == 'http.response.body' and message.get('body'):
                first_body.set()

        path = f'/api/v1/jobs/{job_id}/events'
        with patch.dict(app.state._state, {'jobs': manager}):
            await asyncio.wait_for(app({
                'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                'method': 'GET', 'scheme': 'http', 'path': path, 'raw_path': path.encode(),
                'query_string': b'', 'headers': [], 'client': ('127.0.0.1', 1234),
                'server': ('test', 80), 'root_path': '',
            }, receive, send), timeout=2)
        start = next(message for message in messages if message['type'] == 'http.response.start')
        body = b''.join(message.get('body', b'') for message in messages
                        if message['type'] == 'http.response.body')
        return start['status'], dict(start['headers']), body

    async def test_http_stream_headers_and_terminal_payload(self):
        """SSE MIME·캐시·프록시 버퍼 헤더와 완료 JSON을 실제 ASGI 응답에서 확인한다.

        :return: None.
        """
        record = self.record('completed')
        manager = SimpleNamespace(get=Mock(return_value=record))
        status, headers, body = await self.request(manager)
        self.assertEqual(status, 200)
        self.assertEqual(headers[b'content-type'], b'text/event-stream; charset=utf-8')
        self.assertEqual(headers[b'cache-control'], b'no-cache, no-store')
        self.assertEqual(headers[b'x-accel-buffering'], b'no')
        self.assertEqual(self.payload(body.decode()), record.model_dump(mode='json'))
        manager.get.assert_called_once_with(record.job_id)

    async def test_http_missing_and_unavailable_fail_before_stream_headers(self):
        """없는 ID·만료는 일반 JSON 404, 작업자 미준비는 503으로 먼저 응답한다.

        :return: None.
        """
        for job_id in ('bad-id', '0' * 32):
            status, headers, body = await self.request(SimpleNamespace(get=Mock(side_effect=KeyError)), job_id)
            self.assertEqual(status, 404)
            self.assertIn(b'application/json', headers[b'content-type'])
            self.assertEqual(json.loads(body)['detail'], 'Job not found')
        status, _, _ = await self.request(None)
        self.assertEqual(status, 503)

    async def test_http_disconnect_stops_subscription_without_job_mutation(self):
        """실제 ASGI 연결 해제로 대기 중 스트림이 종료되고 관리자 상태는 유지된다.

        :return: None.
        """
        record = self.record()
        manager = Mock()
        manager.get.return_value = record
        status, _, body = await self.request(manager, disconnect=True)
        self.assertEqual(status, 200)
        self.assertEqual(self.payload(body.decode())['status'], 'queued')
        self.assertEqual(manager.method_calls, [('get', (record.job_id,), {})])
        self.assertEqual(record.status, 'queued')

    async def test_openapi_documents_stream_media_type_and_errors(self):
        """SSE 경로가 OpenAPI에 올바른 미디어 유형과 사전 조회 오류로 노출된다.

        :return: None.
        """
        route = app.openapi()['paths']['/api/v1/jobs/{job_id}/events']['get']
        self.assertEqual(set(route['responses']['200']['content']), {'text/event-stream'})
        self.assertIn('404', route['responses'])
        self.assertIn('503', route['responses'])
