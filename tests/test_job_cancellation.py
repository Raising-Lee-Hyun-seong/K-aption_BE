"""실제 모델 없이 작업 취소의 단계 경계·저장·복구·HTTP 계약을 검증한다."""

import asyncio
import io
from pathlib import Path
from threading import Event
from unittest.mock import patch

from fastapi import UploadFile

from app.services.job_events import job_events
from app.services.jobs import JobCapacityError, JobManager, JobNotReadyError, JobStorageError
from tests import test_jobs


class JobCancellationTests(test_jobs.JobTestCase):
    """기존 작업자와 실제 파일을 사용해 취소가 추론·대기열을 안전하게 정리하는지 확인한다."""

    request = test_jobs.JobApiTests.request

    async def block_stt(self, error=None):
        """STT 스레드를 차단하고 실제 실행 진입을 확인한 뒤 작업과 해제 게이트를 준다.

        :param error: STT 게이트 해제 뒤 발생시킬 예외. None이면 정상 결과를 반환한다.
        :return: 실행 중 작업의 접수 스냅샷과 threading.Event 해제 게이트 튜플.
        """
        entered, release = Event(), Event()
        self.gates.append(release)
        response = self.stt.transcribe.return_value

        def transcribe(path, language):
            """첫 작업의 입력을 보존한 채 STT 실행을 차단한다.

            :param path: 실제 작업 미디어 경로.
            :param language: 영어 인식 코드 en.
            :return: 두 문장으로 구성한 STT 대역 결과.
            :raises Exception: error로 지정한 테스트 예외가 있으면 발생시킨다.
            """
            entered.set()
            release.wait(timeout=5)
            if error is not None:
                raise error
            return response

        self.stt.transcribe.side_effect = transcribe
        accepted = await self.submit()
        await self.wait_gate(entered)
        return accepted, release

    async def wait_gate(self, gate):
        """이벤트 루프를 막지 않고 테스트 추론의 진입 신호를 기다린다.

        :param gate: 작업 스레드가 진입 시 설정하는 threading.Event.
        :return: None.
        :raises TimeoutError: 5초 안에 진입하지 않은 경우.
        """
        async def wait():
            """스레드 진입을 짧은 간격으로 확인한다.

            :return: None.
            """
            while not gate.is_set():
                await asyncio.sleep(0.001)
        await asyncio.wait_for(wait(), timeout=5)

    async def test_queued_cancel_releases_capacity_and_preserves_order(self):
        """대기 취소가 즉시 용량을 반환하고 다른 작업 순서·join 계수를 유지한다.

        :return: None.
        """
        first, release = await self.block_stt()
        second, third = await self.submit(), await self.submit()
        self.assertEqual(self.manager._active, 3)
        cancelled = self.manager.cancel(second.job_id)
        self.assertEqual((cancelled.status, cancelled.stage), ('cancelled', 'cancelled'))
        self.assertEqual(self.manager._active, 2)
        self.assertFalse((self.root / second.job_id / 'input.media').exists())
        fourth = await self.submit()
        joined = asyncio.create_task(self.manager.queue.join())
        await asyncio.sleep(0)
        self.assertFalse(joined.done())
        release.set()
        await asyncio.wait_for(joined, timeout=5)
        self.assertEqual([call.args[0].parent.name for call in self.stt.transcribe.call_args_list],
                         [first.job_id, third.job_id, fourth.job_id])
        self.assertEqual(self.manager._active, 0)
        self.assertEqual(self.manager._queued, set())

    async def test_running_stt_cancel_waits_and_skips_translation(self):
        """STT 중 취소는 입력과 용량을 유지하고 STT 종료 후 번역 없이 끝낸다.

        :return: None.
        """
        accepted, release = await self.block_stt()
        self.manager.max_active = 1
        requested = self.manager.cancel(accepted.job_id)
        self.assertEqual(requested.status, 'cancelling')
        self.assertIsNone(requested.expires_at)
        self.assertTrue((self.root / accepted.job_id / 'input.media').exists())
        self.assertEqual(self.manager._active, 1)
        with self.assertRaises(JobCapacityError):
            await self.submit()
        self.assertEqual(self.manager.cancel(accepted.job_id), requested)
        release.set()
        await self.finish()
        final = self.manager.get(accepted.job_id)
        self.assertEqual(final.status, 'cancelled')
        self.assertIsNone(final.error)
        self.assertIsNone(final.progress_percent)
        self.assertIsNotNone(final.expires_at)
        self.translator.translate.assert_not_called()
        self.assertFalse((self.root / accepted.job_id / 'input.media').exists())
        with self.assertRaises(JobNotReadyError):
            self.manager.result(accepted.job_id)

    async def test_running_translation_cancel_skips_remaining_segments(self):
        """번역 중 취소가 현재 구간까지 기다리고 다음 구간을 실행하지 않는다.

        :return: None.
        """
        entered, release = Event(), Event()
        self.gates.append(release)

        def translate(text, context, max_tokens):
            """현재 번역 구간을 테스트 게이트까지 실행한다.

            :param text: 번역할 현재 영어 문장.
            :param context: 앞 문장의 영어 원문.
            :param max_tokens: 번역 생성 토큰 상한.
            :return: 한국어 대역 번역 사전.
            """
            entered.set()
            release.wait(timeout=5)
            return {'translation': '첫 문장입니다.'}

        self.translator.translate.side_effect = translate
        accepted = await self.submit()
        await self.wait_gate(entered)
        self.assertEqual(self.manager.get(accepted.job_id).stage, 'translating')
        self.manager.cancel(accepted.job_id)
        self.assertTrue((self.root / accepted.job_id / 'input.media').exists())
        release.set()
        await self.finish()
        self.assertEqual(self.translator.translate.call_count, 1)
        self.assertEqual(self.manager.get(accepted.job_id).status, 'cancelled')
        self.assertFalse((self.root / accepted.job_id / 'result.json').exists())

    async def test_cancel_remains_honoured_when_current_stt_fails(self):
        """취소 접수 후 현재 STT가 오류로 끝나도 재추론 없이 취소를 완료한다.

        :return: None.
        """
        accepted, release = await self.block_stt(error=ValueError('private media path'))
        self.manager.cancel(accepted.job_id)
        release.set()
        await self.finish()
        self.assertEqual(self.manager.get(accepted.job_id).status, 'cancelled')
        self.translator.translate.assert_not_called()

    async def test_inflight_upload_cannot_be_cancelled_or_deleted(self):
        """업로드 접수 중에는 취소를 거절하고 복사 스레드의 디렉터리를 보존한다.

        :return: None.
        """
        entered, release = Event(), Event()
        self.gates.append(release)
        original = self.manager._save_input

        def save(upload, path):
            """업로드 복사를 게이트에서 기다린 뒤 원래 검증까지 완료한다.

            :param upload: 읽을 UploadFile.
            :param path: 작업 입력 파일 경로.
            :return: None.
            """
            entered.set()
            release.wait(timeout=5)
            original(upload, path)

        upload = UploadFile(filename='lecture.wav', file=io.BytesIO(self.media()))
        with patch.object(self.manager, '_save_input', save):
            submission = asyncio.create_task(self.manager.submit(upload))
            try:
                await self.wait_gate(entered)
                job_id = next(iter(self.manager.records))
                with self.assertRaises(JobNotReadyError):
                    self.manager.cancel(job_id)
                self.assertTrue((self.root / job_id).is_dir())
                self.assertEqual(self.manager.get(job_id).status, 'queued')
                self.assertEqual(self.manager._active, 1)
            finally:
                release.set()
                accepted = await submission
                await upload.close()
        await self.finish()
        self.assertEqual(self.manager.get(accepted.job_id).status, 'completed')

    async def test_cancellation_request_save_failure_keeps_running_job(self):
        """취소 요청 저장 실패를 성공으로 응답하지 않고 기존 실행 상태를 보존한다.

        :return: None.
        """
        accepted, release = await self.block_stt()
        with patch.object(self.manager, '_write_json', side_effect=OSError('private path')):
            with self.assertRaises(JobStorageError):
                self.manager.cancel(accepted.job_id)
        self.assertEqual(self.manager.get(accepted.job_id).status, 'running')
        self.assertTrue((self.root / accepted.job_id / 'input.media').exists())
        release.set()
        await self.finish()
        self.assertEqual(self.manager.get(accepted.job_id).status, 'completed')

    async def test_queued_cancel_cleanup_failure_is_storage_error(self):
        """입력 삭제 실패는 저장 오류로 표시하고 용량·대기열을 한 번만 정리한다.

        :return: None.
        """
        _, release = await self.block_stt()
        queued = await self.submit()
        original = Path.unlink
        target = self.root / queued.job_id / 'input.media'

        def unlink(path, missing_ok=False):
            """특정 취소 입력의 삭제만 실패시킨다.

            :param path: 삭제할 파일 경로.
            :param missing_ok: 파일 부재를 허용하는 옵션.
            :return: None.
            :raises OSError: 취소 대상 입력을 삭제하려는 경우.
            """
            if path == target:
                raise OSError('private disk error')
            return original(path, missing_ok=missing_ok)

        with patch.object(Path, 'unlink', unlink):
            with self.assertRaises(JobStorageError):
                self.manager.cancel(queued.job_id)
        failed = self.manager.get(queued.job_id)
        self.assertEqual((failed.status, failed.error.code), ('failed', 'storage_error'))
        self.assertEqual(self.manager._active, 1)
        release.set()
        await self.finish()
        self.assertEqual(self.stt.transcribe.call_count, 1)
        self.assertEqual(self.manager._active, 0)

    async def test_running_cancel_save_failure_does_not_stop_worker(self):
        """취소 완료 저장 실패를 공개 저장 오류로 바꾸고 다음 작업은 정상 실행한다.

        :return: None.
        """
        running, release = await self.block_stt()
        queued = await self.submit()
        self.manager.cancel(running.job_id)
        original = self.manager._write_json

        def write(path, payload):
            """취소 완료 상태 기록만 실패시키고 나머지 상태·결과는 저장한다.

            :param path: 상태 또는 결과 JSON 파일 경로.
            :param payload: 직렬화할 상태 또는 결과 사전.
            :return: None.
            :raises OSError: 취소 완료 상태를 기록하려는 경우.
            """
            if payload.get('status') == 'cancelled':
                raise OSError('private disk error')
            original(path, payload)

        with patch.object(self.manager, '_write_json', write):
            release.set()
            await self.finish()
        final = self.manager.get(running.job_id)
        self.assertEqual((final.status, final.error.code), ('failed', 'storage_error'))
        self.assertEqual(self.manager.get(queued.job_id).status, 'completed')
        self.assertEqual(self.manager._active, 0)

    async def test_cancelled_retry_preserves_ttl_elapsed_and_sse_termination(self):
        """취소 재요청이 TTL·경과 시간을 늘리지 않고 SSE 종료 상태로 전달된다.

        :return: None.
        """
        _, release = await self.block_stt()
        queued = await self.submit()
        self.clock.return_value += 10
        cancelled = self.manager.cancel(queued.job_id)
        self.clock.return_value += 10
        self.assertEqual(self.manager.cancel(queued.job_id), cancelled)
        chunks = [chunk async for chunk in job_events(self.manager, cancelled)]
        self.assertEqual(len(chunks), 1)
        self.assertIn('"status":"cancelled"', chunks[0])
        release.set()
        await self.finish()
        self.clock.return_value += self.manager.ttl_seconds
        with self.assertRaises(KeyError):
            self.manager.cancel(queued.job_id)

    async def test_restart_finishes_persisted_cancellation_without_inference(self):
        """강제 종료 직전 저장된 취소 요청은 재시작에서 파일 정리 후 취소 완료된다.

        :return: None.
        """
        accepted = await self.submit()
        await self.finish()
        # 취소 요청 직후 프로세스가 종료된 디스크 스냅샷을 구성한다.
        self.manager._update(accepted.job_id, status='cancelling', stage='transcribing', expires_at=None)
        (self.root / accepted.job_id / 'input.media').write_bytes(b'interrupted input')
        await self.manager.close()
        self.stt.transcribe.reset_mock()
        self.translator.translate.reset_mock()
        self.manager = JobManager(self.root, self.stt, self.translator, clock=self.clock)
        await self.manager.start()
        self.assertEqual(self.manager.get(accepted.job_id).status, 'cancelled')
        self.assertFalse((self.root / accepted.job_id / 'input.media').exists())
        self.assertFalse((self.root / accepted.job_id / 'result.json').exists())
        self.stt.transcribe.assert_not_called()
        self.translator.translate.assert_not_called()

    async def test_shutdown_preserves_cancellation_and_fails_other_waiting_jobs(self):
        """정상 종료 중 취소 요청은 유지하고 나머지 대기 작업은 중단 처리한다.

        :return: None.
        """
        running, release = await self.block_stt()
        cancelled, pending = await self.submit(), await self.submit()
        self.manager.cancel(running.job_id)
        self.manager.cancel(cancelled.job_id)
        closing = asyncio.create_task(self.manager.close())
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(closing, timeout=5)
        self.assertEqual(self.manager.get(running.job_id).status, 'cancelled')
        self.assertEqual(self.manager.get(cancelled.job_id).status, 'cancelled')
        self.assertEqual(self.manager.get(pending.job_id).error.code, 'interrupted')
        self.assertEqual(self.manager._active, 0)
        await asyncio.wait_for(self.manager.queue.join(), timeout=1)

    async def test_http_cancel_running_queued_retry_and_results(self):
        """실행 202·대기 200·재요청·상태 조회와 취소 결과 409 계약을 확인한다.

        :return: None.
        """
        running, release = await self.block_stt()
        queued = await self.submit()
        for record, expected, state in ((running, 202, 'cancelling'), (queued, 200, 'cancelled')):
            for _ in range(2):
                status, body, headers = await self.request('POST', f'/api/v1/jobs/{record.job_id}/cancel')
                self.assertEqual((status, body['status']), (expected, state))
                self.assertEqual(headers[b'cache-control'], b'no-store')
            status, body, _ = await self.request('GET', f'/api/v1/jobs/{record.job_id}')
            self.assertEqual((status, body['status']), (200, state))
            for suffix in ('result', 'subtitles'):
                status, _, _ = await self.request('GET', f'/api/v1/jobs/{record.job_id}/{suffix}')
                self.assertEqual(status, 409)
        release.set()
        await self.finish()
        status, body, _ = await self.request('POST', f'/api/v1/jobs/{running.job_id}/cancel')
        self.assertEqual((status, body['status']), (200, 'cancelled'))

    async def test_http_cancel_missing_finished_and_storage_errors(self):
        """없는 작업 404·종료 작업 409·저장 오류 500을 공개 안내로 반환한다.

        :return: None.
        """
        for job_id in ('bad-id', '0' * 32):
            status, _, _ = await self.request('POST', f'/api/v1/jobs/{job_id}/cancel')
            self.assertEqual(status, 404)
        record = await self.submit()
        await self.finish()
        for state in ('completed', 'failed'):
            self.manager._update(record.job_id, status=state, stage=state)
            status, _, _ = await self.request('POST', f'/api/v1/jobs/{record.job_id}/cancel')
            self.assertEqual(status, 409)
        with patch.object(self.manager, 'cancel', side_effect=JobStorageError('private path')):
            status, body, _ = await self.request('POST', f'/api/v1/jobs/{record.job_id}/cancel')
        self.assertEqual(status, 500)
        self.assertNotIn('private path', str(body))

    async def test_openapi_lists_cancellation_states_and_responses(self):
        """OpenAPI에 취소 응답·상태·단계가 프론트엔드 계약으로 노출된다.

        :return: None.
        """
        spec = test_jobs.app.openapi()
        route = spec['paths']['/api/v1/jobs/{job_id}/cancel']['post']
        for code in ('200', '202', '404', '409', '500', '503'):
            self.assertIn(code, route['responses'])
        fields = spec['components']['schemas']['JobStatusResponse']['properties']
        self.assertIn('cancelling', fields['status']['enum'])
        self.assertIn('cancelled', fields['status']['enum'])
        self.assertIn('cancelled', fields['stage']['enum'])
