"""실제 모델 없이 작업 API의 수명주기·진행률·저장·복구·파일 정리를 검증한다."""
import asyncio
from datetime import timedelta
import io
import json
from pathlib import Path
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4
import wave

from fastapi import UploadFile

from app.main import app, lifespan
from app.schemas import JobStatusResponse
from app.services.jobs import JobCapacityError, JobManager, JobNotReadyError, JobStorageError
from app.services.pipeline import MediaValidationError


class JobTestCase(unittest.IsolatedAsyncioTestCase):
    """테스트마다 독립 저장소와 단일 작업자 및 종료 시 해제할 추론 게이트를 준비한다."""

    async def asyncSetUp(self):
        """임시 저장소·고정 시계·모델 대역으로 작업자를 시작한다.

        :return: None.
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.clock = Mock(return_value=1800000000)
        self.stt = SimpleNamespace(transcribe=Mock(return_value={'segments': [
            {'start': 0, 'end': 1, 'text': 'First.'},
            {'start': 1, 'end': 2, 'text': 'Second.'}]}))
        self.translator = SimpleNamespace(translate=Mock(return_value={'translation': '번역입니다.'}))
        self.manager = JobManager(self.root, self.stt, self.translator, clock=self.clock)
        self.gates = []
        await self.manager.start()

    async def asyncTearDown(self):
        """차단된 추론을 해제하고 작업자를 종료한 뒤 테스트 파일을 삭제한다.

        :return: None.
        """
        for gate in self.gates:
            gate.set()
        try:
            await self.manager.close()
        finally:
            self.temporary.cleanup()

    def media(self) -> bytes:
        """실제 PyAV 사전 검사를 통과하는 짧은 무음 WAV 바이트를 만든다.

        :return: 16kHz·모노·16비트 0.1초 WAV 바이트.
        """
        output = io.BytesIO()
        with wave.open(output, 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b'\x00\x00' * 1600)
        return output.getvalue()

    async def submit(self):
        """유효한 파일을 접수하고 요청 소유 업로드 스트림을 닫는다.

        :return: 접수한 작업의 상태 스냅샷.
        """
        upload = UploadFile(filename='lecture.wav', file=io.BytesIO(self.media()))
        try:
            return await self.manager.submit(upload)
        finally:
            await upload.close()

    async def finish(self):
        """테스트 대기열의 처리가 제한 시간 안에 끝날 때까지 기다린다.

        :return: None.
        :raises TimeoutError: 작업자가 5초 안에 테스트 처리를 끝내지 못할 때.
        """
        await asyncio.wait_for(self.manager.queue.join(), timeout=5)


class JobManagerTests(JobTestCase):
    """백그라운드 상태 전이와 디스크·취소·재시작 경계 동작을 확인한다."""

    async def test_completion_result_and_input_cleanup(self):
        """접수 직후 queued를 반환하고 완료 결과를 저장한 뒤 입력만 삭제한다.

        :return: None.
        """
        accepted = await self.submit()
        self.assertEqual(accepted.status, 'queued')
        with self.assertRaises(JobNotReadyError):
            self.manager.result(accepted.job_id)
        await self.finish()
        status = self.manager.get(accepted.job_id)
        self.assertEqual((status.status, status.progress_percent, status.total_segments),
                         ('completed', 100, 2))
        result = self.manager.result(status.job_id)
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].text, 'First.')
        self.assertFalse((self.root / status.job_id / 'input.media').exists())
        self.assertTrue((self.root / status.job_id / 'result.json').exists())

    async def test_real_progress_counts_partial_translation_failure(self):
        """실제 구간 완료 수로 0·50·100%를 표시하고 부분 실패를 완료 결과에 남긴다.

        :return: None.
        """
        entered = [asyncio.Event(), asyncio.Event()]
        self.gates = [Event(), Event()]
        loop, calls = asyncio.get_running_loop(), []

        def translate(text, context, max_tokens):
            """구간별 게이트에서 기다린 뒤 첫 번역만 검증 실패로 만든다.

            :param text: 현재 영어 원문.
            :param context: 앞 영어 문맥.
            :param max_tokens: 출력 토큰 한도.
            :return: 두 번째 구간의 테스트 번역 사전.
            :raises ValueError: 첫 번째 구간의 검증 실패.
            """
            index = len(calls)
            calls.append((text, context, max_tokens))
            loop.call_soon_threadsafe(entered[index].set)
            self.gates[index].wait(5)
            if index == 0:
                raise ValueError('Rejected output')
            return {'translation': '두 번째 문장입니다.'}

        self.translator.translate.side_effect = translate
        accepted = await self.submit()
        await asyncio.wait_for(entered[0].wait(), 3)
        status = self.manager.get(accepted.job_id)
        self.assertEqual((status.stage, status.progress_percent), ('translating', 0))
        self.gates[0].set()
        await asyncio.wait_for(entered[1].wait(), 3)
        status = self.manager.get(accepted.job_id)
        self.assertEqual((status.completed_segments, status.progress_percent, status.failed_segments), (1, 50, 1))
        self.gates[1].set()
        await self.finish()
        self.assertEqual(self.manager.get(accepted.job_id).status, 'completed')
        result = self.manager.result(accepted.job_id)
        self.assertEqual(result.segments[0].translation_error, 'translation_failed')
        self.assertEqual(calls[1][1], 'First.')

    async def test_capacity_serial_processing_and_graceful_shutdown(self):
        """실행·대기 작업이 한도를 공유하며 종료는 대기를 실패 처리하고 실행을 기다린다.

        :return: None.
        """
        entered, release = asyncio.Event(), Event()
        self.gates.append(release)
        loop = asyncio.get_running_loop()
        original = self.stt.transcribe.return_value

        def transcribe(path, language):
            """실제 스레드에서 종료 게이트를 기다리는 STT 대역이다.

            :param path: 작업 입력 파일 경로.
            :param language: 인식 언어 코드.
            :return: 준비한 STT 구간 사전.
            """
            loop.call_soon_threadsafe(entered.set)
            release.wait(5)
            return original

        self.stt.transcribe.side_effect = transcribe
        self.manager.max_active = 2
        first = await self.submit()
        await asyncio.wait_for(entered.wait(), 3)
        self.assertIsNone(self.manager.get(first.job_id).progress_percent)
        second = await self.submit()
        self.assertEqual(self.manager.get(second.job_id).status, 'queued')
        with self.assertRaises(JobCapacityError):
            await self.submit()
        self.assertEqual(self.stt.transcribe.call_count, 1)
        closing = asyncio.create_task(self.manager.close())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        self.assertEqual(self.manager.get(second.job_id).error.code, 'interrupted')
        self.assertFalse((self.root / second.job_id / 'input.media').exists())
        self.assertTrue((self.root / first.job_id / 'input.media').exists())
        release.set()
        await asyncio.wait_for(closing, 3)
        self.assertEqual(self.manager.get(first.job_id).status, 'completed')

    async def test_empty_speech_and_safe_worker_errors(self):
        """무음은 빈 완료 결과로, 입력·내부 오류는 안전한 실패 코드로 구분한다.

        :return: None.
        """
        self.stt.transcribe.return_value = {'segments': []}
        accepted = await self.submit()
        await self.finish()
        self.assertEqual(self.manager.result(accepted.job_id).segments, [])
        self.translator.translate.assert_not_called()
        for exception, code in ((ValueError('private decoder path'), 'invalid_media'),
                                (RuntimeError('private model prompt'), 'processing_failed')):
            self.stt.transcribe.side_effect = exception
            accepted = await self.submit()
            await self.finish()
            status = self.manager.get(accepted.job_id)
            self.assertEqual(status.error.code, code)
            self.assertNotIn('private', status.model_dump_json())
            self.assertFalse((self.root / status.job_id / 'input.media').exists())

    async def test_recovery_preserves_result_and_fails_unfinished_job(self):
        """재시작 시 완료 결과는 유지하고 이전 미완료 작업은 재추론 없이 실패 처리한다.

        :return: None.
        """
        completed = await self.submit()
        await self.finish()
        interrupted_id = uuid4().hex
        directory = self.root / interrupted_id
        directory.mkdir()
        record = completed.model_copy(update={'job_id': interrupted_id, 'status': 'running',
                                              'stage': 'transcribing'})
        (directory / 'status.json').write_text(record.model_dump_json())
        (directory / 'input.media').write_bytes(self.media())
        expired_id = uuid4().hex
        expired = self.root / expired_id
        expired.mkdir()
        expired_record = record.model_copy(update={
            'job_id': expired_id, 'status': 'completed', 'stage': 'completed',
            'expires_at': self.manager._now() - timedelta(seconds=1),
        })
        (expired / 'status.json').write_text(expired_record.model_dump_json())
        corrupt_id = uuid4().hex
        corrupt = self.root / corrupt_id
        corrupt.mkdir()
        (corrupt / 'status.json').write_text('{broken')
        (corrupt / 'input.media').write_bytes(self.media())
        calls = self.stt.transcribe.call_count
        await self.manager.close()
        self.manager = JobManager(self.root, self.stt, self.translator, clock=self.clock)
        await self.manager.start()
        self.assertEqual(len(self.manager.result(completed.job_id).segments), 2)
        self.assertEqual(self.manager.get(interrupted_id).error.code, 'interrupted')
        self.assertFalse((directory / 'input.media').exists())
        self.assertEqual(self.stt.transcribe.call_count, calls)
        self.assertFalse(expired.exists())
        self.assertEqual(self.manager.get(corrupt_id).error.code, 'storage_error')
        self.assertFalse((corrupt / 'input.media').exists())

    async def test_expiry_and_corrupt_result(self):
        """손상 결과를 실패로 표시하고 보관 기한 이후에는 상태와 파일을 삭제한다.

        :return: None.
        """
        accepted = await self.submit()
        await self.finish()
        (self.root / accepted.job_id / 'result.json').write_text('{invalid')
        with self.assertRaises(JobStorageError):
            self.manager.result(accepted.job_id)
        self.assertEqual(self.manager.get(accepted.job_id).error.code, 'storage_error')
        self.clock.return_value += 86401
        with self.assertRaises(KeyError):
            self.manager.get(accepted.job_id)
        self.assertFalse((self.root / accepted.job_id).exists())
        with self.assertRaises(KeyError):
            self.manager.get('../outside')

    async def test_rejected_upload_and_storage_failure_cleanup(self):
        """잘못된 미디어와 결과 저장 실패에서도 입력을 지우고 작업자는 계속 동작한다.

        :return: None.
        """
        upload = UploadFile(filename='broken.wav', file=io.BytesIO(b'not media'))
        try:
            with self.assertRaises(MediaValidationError):
                await self.manager.submit(upload)
        finally:
            await upload.close()
        self.assertEqual(self.manager.records, {})
        original = self.manager._write_json

        def write(path, payload):
            """결과 저장만 디스크 오류로 실패시키고 상태 저장은 유지한다.

            :param path: 기록할 JSON 경로.
            :param payload: 저장할 응답 사전.
            :return: None.
            :raises OSError: 결과 파일을 저장하려 할 때.
            """
            if path.name == 'result.json':
                raise OSError('private storage path')
            original(path, payload)

        with patch.object(self.manager, '_write_json', side_effect=write):
            accepted = await self.submit()
            await self.finish()
        self.assertEqual(self.manager.get(accepted.job_id).error.code, 'storage_error')
        self.assertFalse((self.root / accepted.job_id / 'input.media').exists())
        next_job = await self.submit()
        await self.finish()
        self.assertEqual(self.manager.get(next_job.job_id).status, 'completed')

    async def test_cancelled_submission_waits_for_copy_before_cleanup(self):
        """접수 취소 시 복사 스레드 종료를 기다린 후 파일·용량을 정리한다.

        :return: None.
        """
        entered, release = asyncio.Event(), Event()
        self.gates.append(release)
        loop = asyncio.get_running_loop()
        original = self.manager._save_input

        def save(upload, path):
            """게이트 해제 후 실제 파일 저장을 진행한다.

            :param upload: 테스트 업로드 객체.
            :param path: 저장할 입력 파일 경로.
            :return: None.
            """
            loop.call_soon_threadsafe(entered.set)
            release.wait(5)
            original(upload, path)

        with patch.object(self.manager, '_save_input', side_effect=save):
            task = asyncio.create_task(self.submit())
            await asyncio.wait_for(entered.wait(), 3)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(self.manager.records, {})
        self.assertEqual(self.manager._active, 0)
        self.assertEqual([p for p in self.root.iterdir() if p.is_dir()], [])

    async def test_store_lock_prevents_second_worker(self):
        """같은 저장소의 두 번째 작업자는 시작하지 못한다.

        :return: None.
        """
        other = JobManager(self.root, self.stt, self.translator)
        with self.assertRaisesRegex(RuntimeError, 'single server worker'):
            await other.start()


class JobApiTests(JobTestCase):
    """실제 ASGI multipart 접수와 조회·오류·OpenAPI 계약을 확인한다."""

    async def request(self, method, path, filename=None, data=None, query=b''):
        """실제 앱에 테스트 요청을 보내 상태·JSON·응답 헤더를 반환한다.

        :param method: GET 또는 POST 요청 메서드.
        :param path: 요청 API 경로.
        :param filename: multipart 업로드 이름. None이면 업로드 본문을 만들지 않는다.
        :param data: 선택적 파일 바이트. 기본값은 유효한 WAV다.
        :param query: URL 인코딩한 쿼리 바이트.
        :return: 상태 코드·JSON 응답·헤더 사전 튜플.
        """
        body, headers = b'', []
        if filename is not None:
            body = (f'--job-boundary\r\nContent-Disposition: form-data; name="file"; '
                    f'filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\n').encode()
            body += (self.media() if data is None else data) + b'\r\n--job-boundary--\r\n'
            headers = [(b'content-type', b'multipart/form-data; boundary=job-boundary')]
        sent, messages = False, []

        async def receive():
            """요청 본문을 한 번 제공한다.

            :return: ASGI 본문 또는 연결 종료 이벤트.
            """
            nonlocal sent
            if sent:
                return {'type': 'http.disconnect'}
            sent = True
            return {'type': 'http.request', 'body': body, 'more_body': False}

        async def send(message):
            """앱에서 전송한 응답 이벤트를 모은다.

            :param message: HTTP 응답 ASGI 이벤트.
            :return: None.
            """
            messages.append(message)

        with patch.dict(app.state._state, {'jobs': self.manager}):
            await app({'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                       'method': method, 'scheme': 'http', 'path': path, 'raw_path': path.encode(),
                       'query_string': query, 'headers': headers, 'client': ('127.0.0.1', 1234),
                       'server': ('test', 80), 'root_path': ''}, receive, send)
        start = next(m for m in messages if m['type'] == 'http.response.start')
        payload = b''.join(m.get('body', b'') for m in messages if m['type'] == 'http.response.body')
        return start['status'], json.loads(payload), dict(start['headers'])

    async def test_create_poll_and_result_contract(self):
        """202·Location 접수 후 상태 조회와 기존 자막 형식 결과를 제공한다.

        :return: None.
        """
        status, record, headers = await self.request('POST', '/api/v1/jobs', 'lecture.wav')
        self.assertEqual(status, 202)
        JobStatusResponse.model_validate(record)
        location = f'/api/v1/jobs/{record["job_id"]}'
        self.assertEqual(headers[b'location'].decode(), location)
        await self.finish()
        status, body, _ = await self.request('GET', location)
        self.assertEqual((status, body['status']), (200, 'completed'))
        status, body, _ = await self.request('GET', location + '/result')
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {'segments'})
        self.assertEqual(len(body['segments']), 2)

    async def test_input_capacity_missing_and_failed_contracts(self):
        """입력 검증·용량·없는 작업·실패 결과의 HTTP 계약을 검증한다.

        :return: None.
        """
        for filename, data, query, expected in (
            ('lecture.txt', None, b'', 415), ('lecture.wav', b'bad', b'', 422),
            ('lecture.wav', None, b'language=ko', 422), (None, None, b'', 422),
        ):
            status, _, _ = await self.request('POST', '/api/v1/jobs', filename, data, query)
            self.assertEqual(status, expected)
        with patch('app.services.jobs.MAX_UPLOAD_BYTES', 4):
            status, _, _ = await self.request('POST', '/api/v1/jobs', 'lecture.wav')
            self.assertEqual(status, 413)
        self.manager.max_active = 0
        status, _, headers = await self.request('POST', '/api/v1/jobs', 'lecture.wav')
        self.assertEqual(status, 503)
        self.assertEqual(headers[b'retry-after'], b'5')
        self.manager.max_active = 4
        status, _, _ = await self.request('GET', '/api/v1/jobs/' + '0' * 32)
        self.assertEqual(status, 404)
        self.stt.transcribe.side_effect = RuntimeError('secret')
        _, accepted, _ = await self.request('POST', '/api/v1/jobs', 'lecture.wav')
        await self.finish()
        status, _, _ = await self.request('GET', f'/api/v1/jobs/{accepted["job_id"]}/result')
        self.assertEqual(status, 409)

    async def test_lifespan_releases_storage_after_model_loading_error(self):
        """모델 시작 오류에서도 저장소 잠금이 해제돼 다음 서버가 시작할 수 있다.

        :return: None.
        """
        await self.manager.close()
        with patch.dict('os.environ', {'KAPTION_JOB_DIR': str(self.root)}), patch(
                'app.main.TranslationService', side_effect=RuntimeError('model failure')):
            with self.assertRaisesRegex(RuntimeError, 'model failure'):
                async with lifespan(app):
                    self.fail('Startup must fail')
        self.manager = JobManager(self.root, self.stt, self.translator, clock=self.clock)
        await self.manager.start()

    async def test_openapi_documents_job_routes_and_statuses(self):
        """작업 접수·상태·결과 경로와 202·409 계약이 OpenAPI에 포함된다.

        :return: None.
        """
        spec = app.openapi()
        create = spec['paths']['/api/v1/jobs']['post']
        result = spec['paths']['/api/v1/jobs/{job_id}/result']['get']
        self.assertIn('202', create['responses'])
        self.assertIn('503', create['responses'])
        self.assertIn('409', result['responses'])
